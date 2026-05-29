import LMIR

extension MetalSourceGenerator {
    public static func generateSparseMoE(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        return [
            generateSparseMoEMonolithic(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                gateKind: gateKind
            ),
            generateSparseMoERouter(
                name: "\(name)_router",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                gateKind: gateKind
            ),
            generateSparseMoEGateUp(
                name: "\(name)_gate_up",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEDown(
                name: "\(name)_down",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
        ].joined(separator: "\n\n")
    }

    private static func generateSparseMoEMonolithic(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expression: String) in weightFormat.readExpression(expression) }
        let storedOutput = bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue("total", weightFormat: weightFormat)
            : "total"
        let routingStore: String
        let softmaxPreparation: String
        switch gateKind {
        case .sigmoidTopK:
            routingStore = """
                    const float weight = 1.0f / (1.0f + exp(-logit));
                    routingWeights[expert] = weight;
                    routingScores[expert] = useExpertBias != 0u
                        ? weight + expertBias[expert]
                        : weight;
            """
            softmaxPreparation = ""
        case .topK:
            routingStore = """
                    routingWeights[expert] = logit;
                    routingScores[expert] = logit;
            """
            softmaxPreparation = """

            if (sgitg == 0 && tiisg == 0) {
                float maxLogit = -INFINITY;
                for (uint expert = 0; expert < expertCount; expert++) {
                    maxLogit = max(maxLogit, routingWeights[expert]);
                }
                float weightSum = 0.0f;
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float weight = exp(routingWeights[expert] - maxLogit);
                    routingWeights[expert] = weight;
                    weightSum += weight;
                }
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float weight = routingWeights[expert] / weightSum;
                    routingWeights[expert] = weight;
                    routingScores[expert] = useExpertBias != 0u
                        ? weight + expertBias[expert]
                        : weight;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """
        case .custom:
            preconditionFailure("Sparse MoE generation does not support custom routing gates")
        }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* routerWeight       [[buffer(1)]],
            device const \(wt)* expertGateUpWeight [[buffer(2)]],
            device const \(wt)* expertDownWeight   [[buffer(3)]],
            device const float* expertBias         [[buffer(4)]],
            device \(bt)* output                   [[buffer(5)]],
            constant uint& inputDimension          [[buffer(6)]],
            constant uint& outputDimension         [[buffer(7)]],
            constant uint& intermediateDimension   [[buffer(8)]],
            constant uint& expertCount             [[buffer(9)]],
            constant uint& expertsPerToken         [[buffer(10)]],
            constant uint& normalizeRoutingWeights [[buffer(11)]],
            constant float& routedScalingFactor    [[buffer(12)]],
            constant uint& useExpertBias           [[buffer(13)]],
            constant uint& sequenceLength          [[buffer(14)]],
            constant uint& inputRowStride          [[buffer(15)]],
            constant uint& outputRowStride         [[buffer(16)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint maxExperts = 128u;
            const uint maxTopK = 8u;
            const uint maxIntermediate = 4096u;
            const uint maxRowsPerThreadgroup = 32u;
            const uint rowsPerThreadgroup = min(
                maxRowsPerThreadgroup,
                max(1u, threadsPerThreadgroup.x / SIMD_WIDTH)
            );
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (seqPos >= sequenceLength) {
                return;
            }
            if (expertCount > maxExperts || expertsPerToken > maxTopK) {
                return;
            }
            if (intermediateDimension > maxIntermediate) {
                return;
            }
            const bool validRow = row < outputDimension;

            threadgroup float routingWeights[128];
            threadgroup float routingScores[128];
            threadgroup uint selectedExperts[8];
            threadgroup float selectedWeights[8];
            threadgroup float activated[4096];

            device const \(bt)* inputRow = input + seqPos * inputRowStride;

            if (sgitg == 0) {
                for (uint expert = 0; expert < expertCount; expert++) {
                    float logit = 0.0f;
                    device const \(wt)* routerRow = routerWeight + expert * inputDimension;
                    for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                        logit += \(readWeight("routerRow[j]")) * float(inputRow[j]);
                    }
                    logit = simd_sum(logit);
                    if (tiisg == 0) {
            \(routingStore)
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            \(softmaxPreparation)

            if (sgitg == 0 && tiisg == 0) {
                float selectedWeightSum = 0.0f;
                for (uint k = 0; k < expertsPerToken; k++) {
                    float bestScore = -INFINITY;
                    uint bestExpert = 0u;
                    for (uint expert = 0; expert < expertCount; expert++) {
                        bool alreadySelected = false;
                        for (uint prev = 0; prev < k; prev++) {
                            alreadySelected = alreadySelected || selectedExperts[prev] == expert;
                        }
                        const float score = alreadySelected ? -INFINITY : routingScores[expert];
                        if (score > bestScore) {
                            bestScore = score;
                            bestExpert = expert;
                        }
                    }
                    const float routingWeight = routingWeights[bestExpert];
                    selectedExperts[k] = bestExpert;
                    selectedWeights[k] = routingWeight;
                    selectedWeightSum += routingWeight;
                }
                for (uint k = 0; k < expertsPerToken; k++) {
                    float weight = selectedWeights[k];
                    if (normalizeRoutingWeights != 0u) {
                        weight = weight / (selectedWeightSum + 1.0e-6f);
                    }
                    selectedWeights[k] = weight * routedScalingFactor;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (!validRow) {
                return;
            }

            float total = 0.0f;
            for (uint k = 0; k < expertsPerToken; k++) {
                const uint expert = selectedExperts[k];
                const float routeWeight = selectedWeights[k];
                device const \(wt)* gateBase = expertGateUpWeight
                    + expert * (2u * intermediateDimension * inputDimension);
                device const \(wt)* upBase = gateBase + intermediateDimension * inputDimension;

                for (uint m = tid; m < intermediateDimension; m += threadsPerThreadgroup.x) {
                    float gate = 0.0f;
                    float up = 0.0f;
                    device const \(wt)* gateRow = gateBase + m * inputDimension;
                    device const \(wt)* upRow = upBase + m * inputDimension;
                    for (uint j = 0; j < inputDimension; j++) {
                        const float x = float(inputRow[j]);
                        gate += \(readWeight("gateRow[j]")) * x;
                        up += \(readWeight("upRow[j]")) * x;
                    }
                    activated[m] = gate * (1.0f / (1.0f + exp(-gate))) * up * routeWeight;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float partial = 0.0f;
                if (validRow) {
                    device const \(wt)* downRow = expertDownWeight
                        + expert * (outputDimension * intermediateDimension)
                        + row * intermediateDimension;
                    for (uint m = tiisg; m < intermediateDimension; m += SIMD_WIDTH) {
                        partial += \(readWeight("downRow[m]")) * activated[m];
                    }
                }
                partial = simd_sum(partial);
                if (validRow && tiisg == 0) {
                    total += partial;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storedOutput));
            }
        }
        """
    }

    private static func generateSparseMoERouter(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expression: String) in weightFormat.readExpression(expression) }
        let routingStore: String
        let softmaxPreparation: String
        switch gateKind {
        case .sigmoidTopK:
            routingStore = """
                    const float weight = 1.0f / (1.0f + exp(-logit));
                    routingWeights[expert] = weight;
                    routingScores[expert] = useExpertBias != 0u
                        ? weight + expertBias[expert]
                        : weight;
            """
            softmaxPreparation = ""
        case .topK:
            routingStore = """
                    routingWeights[expert] = logit;
                    routingScores[expert] = logit;
            """
            softmaxPreparation = """

            if (tiisg == 0) {
                float maxLogit = -INFINITY;
                for (uint expert = 0; expert < expertCount; expert++) {
                    maxLogit = max(maxLogit, routingWeights[expert]);
                }
                float weightSum = 0.0f;
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float weight = exp(routingWeights[expert] - maxLogit);
                    routingWeights[expert] = weight;
                    weightSum += weight;
                }
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float weight = routingWeights[expert] / weightSum;
                    routingWeights[expert] = weight;
                    routingScores[expert] = useExpertBias != 0u
                        ? weight + expertBias[expert]
                        : weight;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """
        case .custom:
            preconditionFailure("Sparse MoE generation does not support custom routing gates")
        }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* routerWeight       [[buffer(1)]],
            device const float* expertBias         [[buffer(2)]],
            device float* moeScratch               [[buffer(3)]],
            constant uint& inputDimension          [[buffer(4)]],
            constant uint& expertCount             [[buffer(5)]],
            constant uint& expertsPerToken         [[buffer(6)]],
            constant uint& normalizeRoutingWeights [[buffer(7)]],
            constant float& routedScalingFactor    [[buffer(8)]],
            constant uint& useExpertBias           [[buffer(9)]],
            constant uint& sequenceLength          [[buffer(10)]],
            constant uint& inputRowStride          [[buffer(11)]],
            constant uint& scratchRowStride        [[buffer(12)]],
            uint seqPos                            [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]]
        ) {
            const uint maxExperts = 128u;
            const uint maxTopK = 8u;
            if (seqPos >= sequenceLength || expertCount > maxExperts || expertsPerToken > maxTopK) {
                return;
            }

            threadgroup float routingWeights[128];
            threadgroup float routingScores[128];

            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;

            for (uint expert = 0; expert < expertCount; expert++) {
                float logit = 0.0f;
                device const \(wt)* routerRow = routerWeight + expert * inputDimension;
                for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                    logit += \(readWeight("routerRow[j]")) * float(inputRow[j]);
                }
                logit = simd_sum(logit);
                if (tiisg == 0) {
        \(routingStore)
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            \(softmaxPreparation)

            if (tiisg == 0) {
                float selectedWeightSum = 0.0f;
                uint selectedExperts[8];
                float selectedWeights[8];
                for (uint k = 0; k < expertsPerToken; k++) {
                    float bestScore = -INFINITY;
                    uint bestExpert = 0u;
                    for (uint expert = 0; expert < expertCount; expert++) {
                        bool alreadySelected = false;
                        for (uint prev = 0; prev < k; prev++) {
                            alreadySelected = alreadySelected || selectedExperts[prev] == expert;
                        }
                        const float score = alreadySelected ? -INFINITY : routingScores[expert];
                        if (score > bestScore) {
                            bestScore = score;
                            bestExpert = expert;
                        }
                    }
                    const float routingWeight = routingWeights[bestExpert];
                    selectedExperts[k] = bestExpert;
                    selectedWeights[k] = routingWeight;
                    selectedWeightSum += routingWeight;
                }
                for (uint k = 0; k < expertsPerToken; k++) {
                    float weight = selectedWeights[k];
                    if (normalizeRoutingWeights != 0u) {
                        weight = weight / (selectedWeightSum + 1.0e-6f);
                    }
                    selectedExpertScratch[k] = float(selectedExperts[k]);
                    selectedWeightScratch[k] = weight * routedScalingFactor;
                }
            }
        }
        """
    }

    private static func generateSparseMoEGateUp(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expression: String) in weightFormat.readExpression(expression) }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const \(wt)* expertGateUpWeight [[buffer(1)]],
            device float* moeScratch               [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& intermediateDimension   [[buffer(4)]],
            constant uint& expertsPerToken         [[buffer(5)]],
            constant uint& sequenceLength          [[buffer(6)]],
            constant uint& inputRowStride          [[buffer(7)]],
            constant uint& scratchRowStride        [[buffer(8)]],
            uint2 gid                              [[thread_position_in_grid]]
        ) {
            const uint seqPos = gid.y;
            const uint flat = gid.x;
            if (seqPos >= sequenceLength) {
                return;
            }
            const uint k = flat / intermediateDimension;
            const uint m = flat - k * intermediateDimension;
            if (k >= expertsPerToken || m >= intermediateDimension) {
                return;
            }

            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;
            device float* activationScratch = scratchRow + 2u * expertsPerToken;

            const uint expert = uint(selectedExpertScratch[k]);
            const float routeWeight = selectedWeightScratch[k];
            device const \(wt)* gateBase = expertGateUpWeight
                + expert * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase = gateBase + intermediateDimension * inputDimension;
            device const \(wt)* gateRow = gateBase + m * inputDimension;
            device const \(wt)* upRow = upBase + m * inputDimension;

            float gate = 0.0f;
            float up = 0.0f;
            for (uint j = 0; j < inputDimension; j++) {
                const float x = float(inputRow[j]);
                gate += \(readWeight("gateRow[j]")) * x;
                up += \(readWeight("upRow[j]")) * x;
            }
            activationScratch[k * intermediateDimension + m] =
                gate * (1.0f / (1.0f + exp(-gate))) * up * routeWeight;
        }
        """
    }

    private static func generateSparseMoEDown(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expression: String) in weightFormat.readExpression(expression) }
        let storedOutput = bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue("total", weightFormat: weightFormat)
            : "total"

        return """
        kernel void \(name)(
            device const float* moeScratch          [[buffer(0)]],
            device const \(wt)* expertDownWeight   [[buffer(1)]],
            device \(bt)* output                   [[buffer(2)]],
            constant uint& outputDimension         [[buffer(3)]],
            constant uint& intermediateDimension   [[buffer(4)]],
            constant uint& expertsPerToken         [[buffer(5)]],
            constant uint& sequenceLength          [[buffer(6)]],
            constant uint& outputRowStride         [[buffer(7)]],
            constant uint& scratchRowStride        [[buffer(8)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (seqPos >= sequenceLength || row >= outputDimension) {
                return;
            }

            device const float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device const float* selectedExpertScratch = scratchRow;
            device const float* activationScratch = scratchRow + 2u * expertsPerToken;

            float total = 0.0f;
            for (uint k = 0; k < expertsPerToken; k++) {
                const uint expert = uint(selectedExpertScratch[k]);
                device const \(wt)* downRow = expertDownWeight
                    + expert * (outputDimension * intermediateDimension)
                    + row * intermediateDimension;
                float partial = 0.0f;
                for (uint m = tiisg; m < intermediateDimension; m += SIMD_WIDTH) {
                    partial += \(readWeight("downRow[m]")) *
                        activationScratch[k * intermediateDimension + m];
                }
                total += simd_sum(partial);
            }
            if (tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storedOutput));
            }
        }
        """
    }
}
