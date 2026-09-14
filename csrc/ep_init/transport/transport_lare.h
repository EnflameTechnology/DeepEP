#ifndef EP_TRANSPORT_LARE_H_
#define EP_TRANSPORT_LARE_H_

#include "transport_resource.h"

template <int SIDE>
class TransportLare : public TransportResource<SIDE> {
  public:
    TransportLare(struct epComm* comm, struct epTopoGraph* graph, int channelId, epTopoPort const& portInfo) :
      TransportResource<SIDE>(comm, graph, channelId, portInfo)
    {
      if (this->comm_->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
       for (int i = 0; i < portInfo.count; i++) {
         lareRoceCtxt_[i] = new LareRoceCtxt(comm, portInfo.efmlPorts[i]);
       }
      }
    }
    virtual ~TransportLare() {
      for (int i = 0; i < this->usedPortsCount_; i++) {
        if (lareRoceCtxt_[i]) {
          delete lareRoceCtxt_[i];
          lareRoceCtxt_[i] = nullptr;
        }
      }
    }
    virtual epResult_t setup() {
      EP_CHECK(TransportResource<SIDE>::setup());
      for (int i = 0; i < this->usedPortsCount_; i++) {
        if (lareRoceCtxt_[i]) {
          EP_CHECK(lareRoceCtxt_[i]->setup());

          uint64_t* basePtr = this->localResource_.buffConnInfo.controlBuff.alignedDevPtr + CONTROL_BUFF_BASE_GET(i);
          uint64_t* stepPtr = basePtr + CONTROL_BUFF_OFFSET_GET(STEP);
          uint64_t* directStepPtr = basePtr + CONTROL_BUFF_OFFSET_GET(DIRECT_STEP);
          uint64_t *flagPtr = directStepPtr + CONTROL_BUFF_OFFSET_GET(FLAG);
          uint64_t *directFlagPtr = directStepPtr + CONTROL_BUFF_OFFSET_GET(DIRECT_FLAG);

          INFO(EP_P2P, "LARE %s c[%d] Setup: r[%d] ctrl[%p] fifo[%p] step[%p] flag[%p] directBuff[%p] directStep[%p] directFlag[%p] sq[%p] p[%u] qp[%u] mac[0x%lx] ip[0x%x]",
                                                              this->sideDesc_[SIDE], this->channelId_, this->comm_->rank,
                                                              this->localResource_.buffConnInfo.controlBuff.alignedPtr,
                                                              this->localResource_.buffConnInfo.fifo.alignedPtr,
                                                              stepPtr,
                                                              flagPtr,
                                                              this->localResource_.buffConnInfo.directBuff.alignedPtr,
                                                              directStepPtr,
                                                              directFlagPtr,
                                                              lareRoceCtxt_[i]->sq_.alignedPtr,
                                                              lareRoceCtxt_[i]->connInfo_.qpInfo.portId,
                                                              lareRoceCtxt_[i]->connInfo_.qpInfo.qpId,
                                                              lareRoceCtxt_[i]->connInfo_.qpConfig.mac,
                                                              lareRoceCtxt_[i]->connInfo_.qpConfig.ip);
        }
      }
      return epSuccess;
    }
    virtual epResult_t generate(struct TransportConnInfo & connInfo) {
      EP_CHECK(TransportResource<SIDE>::generate(connInfo));
      for (int i = 0; i < this->usedPortsCount_; i++) {
        if (lareRoceCtxt_[i]) {
          EP_CHECK(lareRoceCtxt_[i]->get(connInfo.roce[i]));
        }
      }
      return epSuccess;
    }
    virtual epResult_t connect(struct TransportConnInfo const& peerConnInfo, struct epConnector & connector) {
      EP_CHECK(TransportResource<SIDE>::connect(peerConnInfo));
      if (this->comm_->efmlArch >= EFML_DEVICE_ARCH_GCU400) {
        struct epSimpleConnInfo* simple = &connector.conn.hostConnInfoV4Ptr->simple;
        struct epBuffInfo* ll128 = &connector.conn.hostConnInfoV4Ptr->ll128;
        simple->contxt.usedPortsCount = this->usedPortsCount_;
        simple->contxt.transType = TRANSPORT_TYPE_LARE;
        // ensure buffAlignment same w/ GCU 4.0 system
        NEQCHECK(this->comm_->capability.buffAlignment, CONTROL_BUFF_ALIGNMENT);
        if CONSTEXPR (SIDE == SEND) {
          // DeepEP no need any buff for now
          // ll128->bufAddr = (uint64_t)this->peerResource_.fifo.alignedDevPtr;
          // simple->contxt.buffInfo.bufAddr = ll128->bufAddr + this->comm_->buffSizes[EP_PROTO_LL128];
          // simple->contxt.buffInfo.directAddr = (uint64_t)this->peerResource_.directBuff.alignedDevPtr;
          //need once registered memory by using lare master port 0 slave mode for ring ll128.
          // EP_CHECK(lareRoceCtxt_[MASTER_PORTID]->registerMem((void*)ll128->bufAddr,
          //                                                  this->comm_->buffSizes[EP_PROTO_LL128], &ll128->bufAddr));
        } else if CONSTEXPR (SIDE == RECV) {
          //exchanging inplace address by using lare master port 0 slave mode.
          if (this->graph_->pattern == EP_TOPO_PATTERN_MESH) {
            uint64_t outputPA;
            EP_CHECK(lareRoceCtxt_[MASTER_PORTID]->registerMem(
                                 (void*)this->comm_->peerDirectBuffAddrs[this->channelId_][peerConnInfo.rank].alignedDevPtr,
                                             this->comm_->peerDirectBuffAddrs[this->channelId_][peerConnInfo.rank].userSize,
                              (uint64_t*)&outputPA));
            this->comm_->devPeerDirectBuffAddrs[this->channelId_][peerConnInfo.rank] = (uint64_t*)outputPA;
            // configV4.peerDirectAddrs is intentionally NOT filled here.
            // DeepEP kernels (both LL and HT) access per-peer direct addresses via
            // epComm->devPeerDirectBuffAddrs[channel][rank], which is set above.
            // Filling configV4 (EP_MAX_OPS * EP_MAX_WORK_ELEMENTS_P2P entries per channel)
            // is only required for ep collective operations which DeepEP does not use.
            // Skipping this loop avoids O(N * EP_MAX_OPS) memory writes and the associated
            // configV4 memory pressure when scaling to large rank counts (128/256).
          }
          // DeepEP no need any buff for now
          // ll128->bufAddr = (uint64_t)this->localResource_.buffConnInfo.fifo.alignedDevPtr;
          // simple->contxt.buffInfo.bufAddr = ll128->bufAddr + this->comm_->buffSizes[EP_PROTO_LL128];
          // simple->contxt.buffInfo.directAddr = (uint64_t)this->localResource_.buffConnInfo.directBuff.alignedDevPtr;
        }
        for (int p = 0; p < this->usedPortsCount_; ++p) {
          uint64_t *localPtr = this->localResource_.buffConnInfo.controlBuff.alignedDevPtr + CONTROL_BUFF_BASE_GET(p);
          uint64_t *remotePtr = this->peerResource_.controlBuff.alignedDevPtr + CONTROL_BUFF_BASE_GET(p);

          simple->contxt.buffInfo.step[p] = localPtr + CONTROL_BUFF_OFFSET_GET(STEP);
          uint64_t *directStep = localPtr + CONTROL_BUFF_OFFSET_GET(DIRECT_STEP);
          simple->contxt.buffInfo.localFlag[p] = localPtr + CONTROL_BUFF_OFFSET_GET(FLAG);
          simple->contxt.buffInfo.peerFlag[p] = remotePtr + CONTROL_BUFF_OFFSET_GET(FLAG);
          uint64_t *directLocalFlag = localPtr + CONTROL_BUFF_OFFSET_GET(DIRECT_FLAG);
          uint64_t *directPeerFlag = remotePtr + CONTROL_BUFF_OFFSET_GET(DIRECT_FLAG);
          ll128->step[p] = simple->contxt.buffInfo.step[p];

          if (lareRoceCtxt_[p]) {
            EP_CHECK(lareRoceCtxt_[p]->connect(peerConnInfo.roce[p], simple->contxt.localLareInfo[p]));
            //using lare master port 0 slave mode for ring ll128.
            if (p == MASTER_PORTID)
              EP_CHECK(lareRoceCtxt_[p]->registerMem(simple->contxt.buffInfo.peerFlag[p], sizeof(uint64_t),
                                                                                     (uint64_t *)&ll128->peerFlag[p]));
            ll128->localFlag[p] = simple->contxt.buffInfo.localFlag[p];
            INFO(EP_P2P, "LARE %s c[%d] Connect local<r[%d] p[%u] q[%u] addr[%p]> "
                          "peer<r[%d] p[%u], q[%u], mac[0x%08llx], ip[0x%x]> \n"
                          "      simple ==> peerFlag[%p] step[%p] localFlag[%p] buffAddr[0x%012lx]\n"
                          "      simple ==> directPeerFlag[%p] directStep[%p] directLocalFlag[%p] directAddr[0x%012lx]\n"
                          "      ll128  ==> peerFlag[%p] step[%p] localFlag[%p] buffAddr[0x%012lx]",
                                    this->sideDesc_[SIDE], this->channelId_, this->comm_->rank,
                                        simple->contxt.localLareInfo[p].qpInfo.portId,
                                        simple->contxt.localLareInfo[p].qpInfo.qpId,
                                        simple->contxt.localLareInfo[p].sqInfo.baseAddr,
                                        peerConnInfo.rank,
                                        peerConnInfo.roce[p].qpInfo.portId,
                                        peerConnInfo.roce[p].qpInfo.qpId,
                                        peerConnInfo.roce[p].qpConfig.mac,
                                        peerConnInfo.roce[p].qpConfig.ip,
                                        simple->contxt.buffInfo.peerFlag[p],
                                        simple->contxt.buffInfo.step[p], simple->contxt.buffInfo.localFlag[p],
                                        simple->contxt.buffInfo.bufAddr,
                                        directPeerFlag,
                                        directStep,
                                        directLocalFlag,
                                        simple->contxt.buffInfo.directAddr,
                                        ll128->peerFlag,
                                        ll128->step[p], ll128->localFlag[p],
                                        ll128->bufAddr);
          }
        }
      }

      return epSuccess;
    }
    virtual epResult_t teardown() {
     for (int i = 0; i < this->usedPortsCount_; i++) {
       if (lareRoceCtxt_[i]) lareRoceCtxt_[i]->teardown();
     }

      EP_CHECK(TransportResource<SIDE>::teardown());
      return epSuccess;
    }
  private:
    LareRoceCtxt* lareRoceCtxt_[MAX_PORTS_PER_TRUNK] = { nullptr };
};

#endif
