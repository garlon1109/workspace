#ifndef __HOOK_CUTABLE_H__
#define __HOOK_CUTABLE_H__

#include "hook/cuda_subset.h"
#include "hook/macro_common.h"

typedef unsigned char		NvU8;
typedef unsigned int		NvU32;
typedef signed int			NvV32;
typedef unsigned int		NvHandle;
typedef unsigned long int	NvP64;
typedef unsigned long int	NvU64;

#define CU_UUID_CONST static const
#define CU_CHAR(x) (char)((x) & 0xff)
// Define the symbol as exportable to other translation units, and
// initialize the value.  Inner set of parens is necessary because
// "bytes" array needs parens within the struct initializer, which
// also needs parens.  
#define CU_DEFINE_UUID(name, a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)          \
    CU_UUID_CONST CUuuid name =                                                \
    {                                                                          \
      {                                                                        \
        CU_CHAR(a), CU_CHAR((a) >> 8), CU_CHAR((a) >> 16), CU_CHAR((a) >> 24), \
        CU_CHAR(b), CU_CHAR((b) >> 8),                                         \
        CU_CHAR(c), CU_CHAR((c) >> 8),                                         \
        CU_CHAR(d0),                                                           \
        CU_CHAR(d1),                                                           \
        CU_CHAR(d2),                                                           \
        CU_CHAR(d3),                                                           \
        CU_CHAR(d4),                                                           \
        CU_CHAR(d5),                                                           \
        CU_CHAR(d6),                                                           \
        CU_CHAR(d7)                                                            \
      }                                                                        \
    }

CU_DEFINE_UUID(CU_ETID_ToolsRm,
    0x0d180614, 0xd672, 0x47cb, 0xab, 0x6e, 0xb9, 0x63, 0xe5, 0x1b, 0xf4, 0xc6);

CU_DEFINE_UUID(CU_ETID_ToolsModule,
    0xbe3f166e, 0x58b9, 0x4d44, 0x83, 0x5c, 0xe1, 0x82, 0xaf, 0xf1, 0x99, 0x1e);

CU_DEFINE_UUID(CU_ETID_ToolsMemory,
    0x2d43dbbf, 0x3cbf, 0x4a5a, 0x94, 0x5e, 0xb3, 0x40, 0x29, 0xe8, 0x1e, 0x75);

CU_DEFINE_UUID(CU_ETID_ToolsDevice,
    0xe14105b1, 0xc7f7, 0x4ac7, 0x9f, 0x64, 0xf2, 0x23, 0xbe, 0x99, 0xf1, 0xe2);

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId);

template<typename F>
CUresult getFuncFromCUDAExportTable(const CUuuid* uuid, size_t idx, F *f)
{
    const void* exportTable;
    CUresult res = cuGetExportTable(&exportTable, uuid);
    if (res != CUDA_SUCCESS) return res;
    *f = (F)((const uint64_t *)exportTable)[idx];
    return res;
}

#endif