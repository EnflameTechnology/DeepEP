/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <map>
#include <string>

#include "comm.h"
#include "graph.h"

epResult_t epTopoPreset(epComm* comm, struct epTopoGraph** graphs, epTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  // mesh topo
  int meshRanks = localRanks;
  struct epTopoGraph* meshGraph = graphs[EP_ALGO_MESH];
  int meshChannels = meshGraph->nChannels;
  for (int c = 0; c < meshChannels; c++) {
    struct epChannel* channel = comm->channels + c;

    int* meshIntra = meshGraph->intra + c * meshRanks;

    memset(channel->mesh.peerRanks, -1, EP_MAX_LOCAL_RANKS * sizeof(int));
    int* peerRank = channel->mesh.peerRanks;
    for (int i = 0; i < meshRanks; i++) {
      if (meshIntra[i] == rank) {
        // topoRanks->meshRecv[c * meshRanks + i] = -1;
        // topoRanks->meshSend[c * meshRanks + i] = -1;
        channel->mesh.peerRanks[meshRanks - 1] = -1;
      } else {
        // topoRanks->meshRecv[c * meshRanks + i] = meshIntra[i];
        // topoRanks->meshSend[c * meshRanks + i] = meshIntra[i];
        *peerRank= meshIntra[i];
        peerRank++;
      }
    }

    topoRanks->meshRecv[c] = meshIntra[0];
    topoRanks->meshSend[c] = meshIntra[meshRanks-1];

    /* Shift by intraRank so that don't send to same peer simultaneously.
    * for example: assume nranks is 4
    * rank 0: [1 2 3] -> [1 2 3]
    * rank 1: [0 2 3] -> [2 3 0]
    * rank 2: [0 1 3] -> [3 0 1]
    * rank 3: [0 1 2] -> [0 1 2]
    */
    leftRotate(channel->mesh.peerRanks, meshRanks - 1, rank % meshRanks);
  }

  // Duplicate channels
  struct epChannel* channel0 = comm->channels;
  struct epChannel* channel1 = channel0 + nChannels;
  int realBlocks = std::min(nChannels, MAXCHANNELS-nChannels);
  if (realBlocks > 0)
    memcpy(channel1, channel0, realBlocks * sizeof(struct epChannel));

  return epSuccess;
}