/*************************************************************************
 * Copyright (c) 2023, ENFLAME CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef EP_ARGCHECK_H_
#define EP_ARGCHECK_H_

#include "core.h"

epResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname);
epResult_t ArgsCheck(struct epInfo* info);

#endif
