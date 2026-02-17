#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"
#include <stddef.h>
#include <stdio.h>

int main() {
  printf("=== NVOFA Struct Offsets & Sizes ===\n");
  printf("sizeof(NV_OF_INIT_PARAMS): %zu\n", sizeof(NV_OF_INIT_PARAMS));
  printf("offsetof(width): %zu\n", offsetof(NV_OF_INIT_PARAMS, width));
  printf("offsetof(height): %zu\n", offsetof(NV_OF_INIT_PARAMS, height));
  printf("offsetof(outGridSize): %zu\n",
         offsetof(NV_OF_INIT_PARAMS, outGridSize));
  printf("offsetof(mode): %zu\n", offsetof(NV_OF_INIT_PARAMS, mode));
  printf("offsetof(perfLevel): %zu\n", offsetof(NV_OF_INIT_PARAMS, perfLevel));
  printf("offsetof(enableExternalHints): %zu\n",
         offsetof(NV_OF_INIT_PARAMS, enableExternalHints));
  printf("offsetof(enableOutputCost): %zu\n",
         offsetof(NV_OF_INIT_PARAMS, enableOutputCost));
  printf("offsetof(hPrivData): %zu\n", offsetof(NV_OF_INIT_PARAMS, hPrivData));

  printf("\n");
  printf("sizeof(NV_OF_CUDA_API_FUNCTION_LIST): %zu\n",
         sizeof(NV_OF_CUDA_API_FUNCTION_LIST));
  printf("offsetof(nvOFInit): %zu\n",
         offsetof(NV_OF_CUDA_API_FUNCTION_LIST, nvOFInit));
  printf("offsetof(nvOFExecute): %zu\n",
         offsetof(NV_OF_CUDA_API_FUNCTION_LIST, nvOFExecute));
  printf("offsetof(nvOFDestroy): %zu\n",
         offsetof(NV_OF_CUDA_API_FUNCTION_LIST, nvOFDestroy));
  printf("offsetof(nvOFGetCaps): %zu\n",
         offsetof(NV_OF_CUDA_API_FUNCTION_LIST, nvOFGetCaps));

  printf("\n");
  printf("sizeof(NV_OF_EXECUTE_INPUT_PARAMS): %zu\n",
         sizeof(NV_OF_EXECUTE_INPUT_PARAMS));
  printf("offsetof(inputFrame): %zu\n",
         offsetof(NV_OF_EXECUTE_INPUT_PARAMS, inputFrame));
  printf("offsetof(referenceFrame): %zu\n",
         offsetof(NV_OF_EXECUTE_INPUT_PARAMS, referenceFrame));
  printf("offsetof(disableTemporalHints): %zu\n",
         offsetof(NV_OF_EXECUTE_INPUT_PARAMS, disableTemporalHints));

  printf("\n");
  printf("sizeof(NV_OF_EXECUTE_OUTPUT_PARAMS): %zu\n",
         sizeof(NV_OF_EXECUTE_OUTPUT_PARAMS));
  printf("offsetof(outputFlowBuffer): %zu\n",
         offsetof(NV_OF_EXECUTE_OUTPUT_PARAMS, outputBuffer));
  printf("offsetof(outputCostBuffer): %zu\n",
         offsetof(NV_OF_EXECUTE_OUTPUT_PARAMS, outputCostBuffer));

  return 0;
}
