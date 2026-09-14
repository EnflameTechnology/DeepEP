/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_CHANNEL_H_
#define EP_CHANNEL_H_
#include "comm.h"

epResult_t initChannel(struct epComm* comm, int channelid);
epResult_t freeChannel(efmlDeviceArchitecture_t efmlArch, struct epChannel* channel, int nRanks);

#endif
