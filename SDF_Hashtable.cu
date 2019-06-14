#include <cuda.h>
#include <cuda_runtime_api.h>
#include "SDF_Hashtable.h"

#define FREE_ENTRY -2
int numBuckets = 20000;

__device__
uint computeHashPos(const int3 pos) {
  const int p0 = 73856093;
  const int p1 = 19349669;
  const int p2 = 83492791;

  int res = ((pos.x * p0)^(pos.y * p1)^(pos.z * p2))%numBuckets;
  if (res < 0) res += numBuckets;
  return (uint)res;
}

__device__
void deleteHashEntry(HashEntry& hashEntry)  {
  hashEntry.pos = int3{0};
  hashEntry.offset = 0;
  hashEntry.ptr = FREE_ENTRY;
}

//TODO : fill these
__device__
void insertVertexIntoHash(const float4* verts,...)  {
}

extern "C" void alloc(const float4x4& deltaT, const float4* verts, uint* bitmask) {

  uint hashPos = 0;
}
