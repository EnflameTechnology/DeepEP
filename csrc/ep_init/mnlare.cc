/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#include "mnlare.h"
#include "transport.h"

// Determine if MNLARE support is available
epResult_t epMnlareCheck(struct epComm* comm) {
  // Check that all ranks have initialized the fabric fully
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].gcuInfo.fabricInfo.state != EFML_GCU_FABRIC_STATE_COMPLETED) return epSuccess;
  }

  // Determine our MNLARE domain/clique
  comm->clique.id = comm->peerInfo[comm->rank].gcuInfo.fabricInfo.cliqueId;
  efmlGcuFabricInfoV_t *myFabricInfo = &comm->peerInfo[comm->rank].gcuInfo.fabricInfo;
  for (int i = 0; i < comm->nRanks; i++) {
    efmlGcuFabricInfoV_t *peerFabricInfo = &comm->peerInfo[i].gcuInfo.fabricInfo;
    // Check if the cluster UUID and cliqueId match
    // A zero UUID means we don't have MNLARE fabric info - disable MNLARE
    //if ((((long *)&peerFabricInfo->clusterUuid)[0]|((long *)peerFabricInfo->clusterUuid)[1]) == 0) return epSuccess;
    if ((memcmp(myFabricInfo->clusterUuid, peerFabricInfo->clusterUuid, EFML_GCU_FABRIC_UUID_LEN) == 0) &&
        (myFabricInfo->cliqueId == peerFabricInfo->cliqueId)) {
      if (i == comm->rank) {
        comm->cliqueRank = comm->clique.size;
      }
      comm->clique.ranks[comm->clique.size++] = i;
    }
  }

  // No MNLARE clique found
  if (comm->clique.size <= 1) return epSuccess;

  comm->MNLARE = 1;
  INFO(EP_INIT, "MNLARE %d cliqueId %x cliqueSize %d cliqueRank %d", comm->MNLARE, comm->clique.id, comm->clique.size, comm->cliqueRank);
  return epSuccess;
}