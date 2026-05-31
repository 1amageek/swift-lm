import LMIR

extension MetalSourceGenerator {
    public static func generateSparseMoE(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        if weightFormat.isQuantized {
            return [
                generateSparseMoERouterParallel(
                    name: "\(name)_router_parallel",
                    bufferPrecision: bufferPrecision,
                    weightFormat: weightFormat,
                    gateKind: gateKind
                ),
                generateSparseMoERouterScores(
                    name: "\(name)_router_scores",
                    bufferPrecision: bufferPrecision,
                    weightFormat: weightFormat,
                    gateKind: gateKind
                ),
                generateSparseMoERouterSelect(
                    name: "\(name)_router_select",
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
            generateSparseMoERouterParallel(
                name: "\(name)_router_parallel",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                gateKind: gateKind
            ),
            generateSparseMoERouterScores(
                name: "\(name)_router_scores",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                gateKind: gateKind
            ),
            generateSparseMoERouterSelect(
                name: "\(name)_router_select",
                gateKind: gateKind
            ),
            generateSparseMoEGateUp(
                name: "\(name)_gate_up",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEGateUpPacked4(
                name: "\(name)_gate_up_packed4",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEGateUpRow2Packed4(
                name: "\(name)_gate_up_row2_packed4",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEGateUpPacked8(
                name: "\(name)_gate_up_packed8",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEGateUpSplit2(
                name: "\(name)_gate_up_split2",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEDown(
                name: "\(name)_down",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEDownPacked4(
                name: "\(name)_down_packed4",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEDownPacked8(
                name: "\(name)_down_packed8",
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            ),
            generateSparseMoEDownSplit2(
                name: "\(name)_down_split2",
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

    private static func generateSparseMoERouterParallel(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        if weightFormat.isQuantized {
            return generateSparseMoEQuantizedRouterParallel(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                gateKind: gateKind
            )
        }
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
                routingScores[expert] = useExpertBias != 0u ? expertBias[expert] : 0.0f;
            """
            softmaxPreparation = """

            if (tid == 0u) {
                float maxLogit = -INFINITY;
                for (uint expert = 0; expert < expertCount; expert++) {
                    maxLogit = max(maxLogit, routingWeights[expert]);
                }
                float weightSum = 0.0f;
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float scoreBias = routingScores[expert];
                    const float weight = exp(routingWeights[expert] - maxLogit);
                    routingWeights[expert] = weight;
                    routingScores[expert] = scoreBias;
                    weightSum += weight;
                }
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float scoreBias = routingScores[expert];
                    const float weight = routingWeights[expert] / weightSum;
                    routingWeights[expert] = weight;
                    routingScores[expert] = weight + scoreBias;
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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint maxExperts = 128u;
            const uint maxTopK = 8u;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint seqPos = gid.x;
            if (seqPos >= sequenceLength
                || expertCount > maxExperts
                || expertsPerToken > maxTopK
                || expertCount > simdgroupsPerThreadgroup) {
                return;
            }

            threadgroup float routingWeights[128];
            threadgroup float routingScores[128];

            const uint expert = sgitg;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;

            if (expert < expertCount) {
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

            if (tid == 0u) {
                float selectedWeightSum = 0.0f;
                uint selectedExperts[8];
                float selectedWeights[8];
                for (uint k = 0; k < expertsPerToken; k++) {
                    float bestScore = -INFINITY;
                    uint bestExpert = 0u;
                    for (uint expertIndex = 0; expertIndex < expertCount; expertIndex++) {
                        bool alreadySelected = false;
                        for (uint prev = 0; prev < k; prev++) {
                            alreadySelected = alreadySelected || selectedExperts[prev] == expertIndex;
                        }
                        const float score = alreadySelected ? -INFINITY : routingScores[expertIndex];
                        if (score > bestScore) {
                            bestScore = score;
                            bestExpert = expertIndex;
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

    private static func generateSparseMoERouterScores(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        if weightFormat.isQuantized {
            return generateSparseMoEQuantizedRouterScores(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                gateKind: gateKind
            )
        }
        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
        let readWeight = { (expression: String) in weightFormat.readExpression(expression) }
        let routingStore: String
        switch gateKind {
        case .sigmoidTopK:
            routingStore = """
                const float weight = 1.0f / (1.0f + exp(-logit));
                routingWeightsScratch[expert] = weight;
                routingScoresScratch[expert] = useExpertBias != 0u
                    ? weight + expertBias[expert]
                    : weight;
            """
        case .topK:
            routingStore = """
                routingWeightsScratch[expert] = logit;
                routingScoresScratch[expert] = useExpertBias != 0u ? expertBias[expert] : 0.0f;
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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]]
        ) {
            const uint maxExperts = 128u;
            const uint seqPos = gid.y;
            const uint expert = gid.x;
            if (seqPos >= sequenceLength || expert >= expertCount || expertCount > maxExperts) {
                return;
            }

            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* routingWeightsScratch = scratchRow + 2u * expertsPerToken;
            device float* routingScoresScratch = routingWeightsScratch + maxExperts;

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
        """
    }

    private static func generateSparseMoERouterSelect(
        name: String,
        gateKind: MoEGateKind
    ) -> String {
        let softmaxPreparation: String
        switch gateKind {
        case .sigmoidTopK:
            softmaxPreparation = ""
        case .topK:
            softmaxPreparation = """
                float maxLogit = -INFINITY;
                for (uint expert = 0; expert < expertCount; expert++) {
                    maxLogit = max(maxLogit, routingWeightsScratch[expert]);
                }
                float weightSum = 0.0f;
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float scoreBias = routingScoresScratch[expert];
                    const float weight = exp(routingWeightsScratch[expert] - maxLogit);
                    routingWeightsScratch[expert] = weight;
                    routingScoresScratch[expert] = scoreBias;
                    weightSum += weight;
                }
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float scoreBias = routingScoresScratch[expert];
                    const float weight = routingWeightsScratch[expert] / weightSum;
                    routingWeightsScratch[expert] = weight;
                    routingScoresScratch[expert] = weight + scoreBias;
                }
            """
        case .custom:
            preconditionFailure("Sparse MoE generation does not support custom routing gates")
        }

        return """
        kernel void \(name)(
            device float* moeScratch               [[buffer(0)]],
            constant uint& expertCount             [[buffer(1)]],
            constant uint& expertsPerToken         [[buffer(2)]],
            constant uint& normalizeRoutingWeights [[buffer(3)]],
            constant float& routedScalingFactor    [[buffer(4)]],
            constant uint& sequenceLength          [[buffer(5)]],
            constant uint& scratchRowStride        [[buffer(6)]],
            uint seqPos                            [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]]
        ) {
            const uint maxExperts = 128u;
            const uint maxTopK = 8u;
            if (tid != 0u || seqPos >= sequenceLength || expertCount > maxExperts || expertsPerToken > maxTopK) {
                return;
            }

            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;
            device float* routingWeightsScratch = scratchRow + 2u * expertsPerToken;
            device float* routingScoresScratch = routingWeightsScratch + maxExperts;

        \(softmaxPreparation)
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
                    const float score = alreadySelected ? -INFINITY : routingScoresScratch[expert];
                    if (score > bestScore) {
                        bestScore = score;
                        bestExpert = expert;
                    }
                }
                const float routingWeight = routingWeightsScratch[bestExpert];
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
        """
    }

    private static func generateSparseMoEGateUp(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        if weightFormat.isQuantized {
            return generateSparseMoEQuantizedGateUp(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }
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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint seqPos = gid.y;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint flat = gid.x * simdgroupsPerThreadgroup + sgitg;
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
            device float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            const uint expert = uint(selectedExpertScratch[k]);
            const float routeWeight = selectedWeightScratch[k];
            device const \(wt)* gateBase = expertGateUpWeight
                + expert * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase = gateBase + intermediateDimension * inputDimension;
            device const \(wt)* gateRow = gateBase + m * inputDimension;
            device const \(wt)* upRow = upBase + m * inputDimension;

            float gate = 0.0f;
            float up = 0.0f;
            for (uint j = tiisg; j < inputDimension; j += SIMD_WIDTH) {
                const float x = float(inputRow[j]);
                gate += \(readWeight("gateRow[j]")) * x;
                up += \(readWeight("upRow[j]")) * x;
            }
            gate = simd_sum(gate);
            up = simd_sum(up);
            if (tiisg == 0) {
                activationScratch[k * intermediateDimension + m] =
                    gate * (1.0f / (1.0f + exp(-gate))) * up * routeWeight;
            }
        }
        """
    }

    private static func generateSparseMoEDown(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        if weightFormat.isQuantized {
            return generateSparseMoEQuantizedDown(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }
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
            device const float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

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

    private static func generateSparseMoEQuantizedRouterParallel(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        let bt = bufferPrecision.metalType
        let weightsPerBlock = weightFormat.weightsPerBlock
        let bytesPerBlock = weightFormat.bytesPerBlock
        let readWeight = quantizedPerWeightExpression(weightFormat)
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
                routingScores[expert] = useExpertBias != 0u ? expertBias[expert] : 0.0f;
            """
            softmaxPreparation = """

            if (tid == 0u) {
                float maxLogit = -INFINITY;
                for (uint expert = 0; expert < expertCount; expert++) {
                    maxLogit = max(maxLogit, routingWeights[expert]);
                }
                float weightSum = 0.0f;
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float scoreBias = routingScores[expert];
                    const float weight = exp(routingWeights[expert] - maxLogit);
                    routingWeights[expert] = weight;
                    routingScores[expert] = scoreBias;
                    weightSum += weight;
                }
                for (uint expert = 0; expert < expertCount; expert++) {
                    const float scoreBias = routingScores[expert];
                    const float weight = routingWeights[expert] / weightSum;
                    routingWeights[expert] = weight;
                    routingScores[expert] = weight + scoreBias;
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
            device const uchar* routerWeight       [[buffer(1)]],
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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tid                               [[thread_index_in_threadgroup]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            const uint maxExperts = 128u;
            const uint maxTopK = 8u;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint seqPos = gid.x;
            if (seqPos >= sequenceLength
                || expertCount > maxExperts
                || expertsPerToken > maxTopK
                || expertCount > simdgroupsPerThreadgroup) {
                return;
            }

            threadgroup float routingWeights[128];
            threadgroup float routingScores[128];

            const uint expert = sgitg;
            const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;

            if (expert < expertCount) {
                float logit = 0.0f;
                device const uchar* routerRow = routerWeight + expert * blocksPerRow * BYTES_PER_BLOCK;
                for (uint blockIndex = 0; blockIndex < blocksPerRow; blockIndex++) {
                    device const uchar* block = routerRow + blockIndex * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    for (uint q = tiisg; q < WEIGHTS_PER_BLOCK; q += SIMD_WIDTH) {
                        const uint j = blockIndex * WEIGHTS_PER_BLOCK + q;
                        logit += \(readWeight) * float(inputRow[j]);
                    }
                }
                logit = simd_sum(logit);
                if (tiisg == 0) {
        \(routingStore)
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            \(softmaxPreparation)

            if (tid == 0u) {
                float selectedWeightSum = 0.0f;
                uint selectedExperts[8];
                float selectedWeights[8];
                for (uint k = 0; k < expertsPerToken; k++) {
                    float bestScore = -INFINITY;
                    uint bestExpert = 0u;
                    for (uint expertIndex = 0; expertIndex < expertCount; expertIndex++) {
                        bool alreadySelected = false;
                        for (uint prev = 0; prev < k; prev++) {
                            alreadySelected = alreadySelected || selectedExperts[prev] == expertIndex;
                        }
                        const float score = alreadySelected ? -INFINITY : routingScores[expertIndex];
                        if (score > bestScore) {
                            bestScore = score;
                            bestExpert = expertIndex;
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

    private static func generateSparseMoEQuantizedRouterScores(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat,
        gateKind: MoEGateKind
    ) -> String {
        let bt = bufferPrecision.metalType
        let weightsPerBlock = weightFormat.weightsPerBlock
        let bytesPerBlock = weightFormat.bytesPerBlock
        let readWeight = quantizedPerWeightExpression(weightFormat)
        let routingStore: String
        switch gateKind {
        case .sigmoidTopK:
            routingStore = """
                const float weight = 1.0f / (1.0f + exp(-logit));
                routingWeightsScratch[expert] = weight;
                routingScoresScratch[expert] = useExpertBias != 0u
                    ? weight + expertBias[expert]
                    : weight;
            """
        case .topK:
            routingStore = """
                routingWeightsScratch[expert] = logit;
                routingScoresScratch[expert] = useExpertBias != 0u ? expertBias[expert] : 0.0f;
            """
        case .custom:
            preconditionFailure("Sparse MoE generation does not support custom routing gates")
        }

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const uchar* routerWeight       [[buffer(1)]],
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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]]
        ) {
            const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            const uint maxExperts = 128u;
            const uint seqPos = gid.y;
            const uint expert = gid.x;
            if (seqPos >= sequenceLength || expert >= expertCount || expertCount > maxExperts) {
                return;
            }

            const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* routingWeightsScratch = scratchRow + 2u * expertsPerToken;
            device float* routingScoresScratch = routingWeightsScratch + maxExperts;

            float logit = 0.0f;
            device const uchar* routerRow = routerWeight + expert * blocksPerRow * BYTES_PER_BLOCK;
            for (uint blockIndex = 0; blockIndex < blocksPerRow; blockIndex++) {
                device const uchar* block = routerRow + blockIndex * BYTES_PER_BLOCK;
                float scale = float(*(device const half*)(block));
                float zero = float(*(device const half*)(block + 2));
                device const uchar* qs = block + 4;
                for (uint q = tiisg; q < WEIGHTS_PER_BLOCK; q += SIMD_WIDTH) {
                    const uint j = blockIndex * WEIGHTS_PER_BLOCK + q;
                    logit += \(readWeight) * float(inputRow[j]);
                }
            }
            logit = simd_sum(logit);
            if (tiisg == 0) {
        \(routingStore)
            }
        }
        """
    }

    private static func generateSparseMoEQuantizedGateUp(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let weightsPerBlock = weightFormat.weightsPerBlock
        let bytesPerBlock = weightFormat.bytesPerBlock
        let readWeight = quantizedPerWeightExpression(weightFormat)

        return """
        kernel void \(name)(
            device const \(bt)* input              [[buffer(0)]],
            device const uchar* expertGateUpWeight [[buffer(1)]],
            device float* moeScratch               [[buffer(2)]],
            constant uint& inputDimension          [[buffer(3)]],
            constant uint& intermediateDimension   [[buffer(4)]],
            constant uint& expertsPerToken         [[buffer(5)]],
            constant uint& sequenceLength          [[buffer(6)]],
            constant uint& inputRowStride          [[buffer(7)]],
            constant uint& scratchRowStride        [[buffer(8)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            const uint seqPos = gid.y;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint flat = gid.x * simdgroupsPerThreadgroup + sgitg;
            if (seqPos >= sequenceLength) {
                return;
            }
            const uint k = flat / intermediateDimension;
            const uint m = flat - k * intermediateDimension;
            if (k >= expertsPerToken || m >= intermediateDimension) {
                return;
            }

            const uint blocksPerRow = inputDimension / WEIGHTS_PER_BLOCK;
            const uint bytesPerRow = blocksPerRow * BYTES_PER_BLOCK;
            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;
            device float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            const uint expert = uint(selectedExpertScratch[k]);
            const float routeWeight = selectedWeightScratch[k];
            device const uchar* gateBase = expertGateUpWeight
                + expert * (2u * intermediateDimension * bytesPerRow);
            device const uchar* upBase = gateBase + intermediateDimension * bytesPerRow;
            device const uchar* gateRow = gateBase + m * bytesPerRow;
            device const uchar* upRow = upBase + m * bytesPerRow;

            float gate = 0.0f;
            float up = 0.0f;
            for (uint blockIndex = 0; blockIndex < blocksPerRow; blockIndex++) {
                device const uchar* gateBlock = gateRow + blockIndex * BYTES_PER_BLOCK;
                device const uchar* upBlock = upRow + blockIndex * BYTES_PER_BLOCK;
                float scale = float(*(device const half*)(gateBlock));
                float zero = float(*(device const half*)(gateBlock + 2));
                device const uchar* qs = gateBlock + 4;
                for (uint q = tiisg; q < WEIGHTS_PER_BLOCK; q += SIMD_WIDTH) {
                    const uint j = blockIndex * WEIGHTS_PER_BLOCK + q;
                    gate += \(readWeight) * float(inputRow[j]);
                }
                scale = float(*(device const half*)(upBlock));
                zero = float(*(device const half*)(upBlock + 2));
                qs = upBlock + 4;
                for (uint q = tiisg; q < WEIGHTS_PER_BLOCK; q += SIMD_WIDTH) {
                    const uint j = blockIndex * WEIGHTS_PER_BLOCK + q;
                    up += \(readWeight) * float(inputRow[j]);
                }
            }
            gate = simd_sum(gate);
            up = simd_sum(up);
            if (tiisg == 0) {
                activationScratch[k * intermediateDimension + m] =
                    gate * (1.0f / (1.0f + exp(-gate))) * up * routeWeight;
            }
        }
        """
    }

    private static func generateSparseMoEQuantizedDown(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bt = bufferPrecision.metalType
        let weightsPerBlock = weightFormat.weightsPerBlock
        let bytesPerBlock = weightFormat.bytesPerBlock
        let readWeight = quantizedPerWeightExpression(weightFormat)
        let storedOutput = bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue("total", weightFormat: .float16)
            : "total"

        return """
        kernel void \(name)(
            device const float* moeScratch          [[buffer(0)]],
            device const uchar* expertDownWeight   [[buffer(1)]],
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
            const uint WEIGHTS_PER_BLOCK = \(weightsPerBlock);
            const uint BYTES_PER_BLOCK = \(bytesPerBlock);
            const uint rowsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint row = gid.x * rowsPerThreadgroup + sgitg;
            const uint seqPos = gid.y;
            if (seqPos >= sequenceLength || row >= outputDimension) {
                return;
            }

            const uint blocksPerRow = intermediateDimension / WEIGHTS_PER_BLOCK;
            const uint bytesPerRow = blocksPerRow * BYTES_PER_BLOCK;
            device const float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device const float* selectedExpertScratch = scratchRow;
            device const float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            float total = 0.0f;
            for (uint k = 0; k < expertsPerToken; k++) {
                const uint expert = uint(selectedExpertScratch[k]);
                device const uchar* downRow = expertDownWeight
                    + expert * (outputDimension * bytesPerRow)
                    + row * bytesPerRow;
                device const float* activatedRow = activationScratch + k * intermediateDimension;
                float partial = 0.0f;
                for (uint blockIndex = 0; blockIndex < blocksPerRow; blockIndex++) {
                    device const uchar* block = downRow + blockIndex * BYTES_PER_BLOCK;
                    float scale = float(*(device const half*)(block));
                    float zero = float(*(device const half*)(block + 2));
                    device const uchar* qs = block + 4;
                    for (uint q = tiisg; q < WEIGHTS_PER_BLOCK; q += SIMD_WIDTH) {
                        const uint m = blockIndex * WEIGHTS_PER_BLOCK + q;
                        partial += \(readWeight) * activatedRow[m];
                    }
                }
                total += simd_sum(partial);
            }
            if (tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storedOutput));
            }
        }
        """
    }

    private static func quantizedPerWeightExpression(_ weightFormat: WeightFormat) -> String {
        guard let expression = weightFormat.perWeightReadExpression(
            blocksVar: "qs",
            weightIndexVar: "q"
        ) else {
            fatalError(
                "Sparse MoE quantized generation requires perWeightReadExpression for \(weightFormat.schemeIdentifier)"
            )
        }
        return expression
    }

    private static func generateSparseMoEGateUpSplit2(
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
            threadgroup float* partials            [[threadgroup(0)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint seqPos = gid.y;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint splitCount = 2u;
            if (simdgroupsPerThreadgroup < splitCount || seqPos >= sequenceLength) {
                return;
            }
            const uint rowsPerThreadgroup = max(1u, simdgroupsPerThreadgroup / splitCount);
            const uint localRow = sgitg / splitCount;
            const uint split = sgitg - localRow * splitCount;
            const uint flat = gid.x * rowsPerThreadgroup + localRow;
            const uint k = flat / intermediateDimension;
            const uint m = flat - k * intermediateDimension;
            if (k >= expertsPerToken || m >= intermediateDimension) {
                return;
            }

            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;
            device float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            const uint expert = uint(selectedExpertScratch[k]);
            const float routeWeight = selectedWeightScratch[k];
            device const \(wt)* gateBase = expertGateUpWeight
                + expert * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase = gateBase + intermediateDimension * inputDimension;
            device const \(wt)* gateRow = gateBase + m * inputDimension;
            device const \(wt)* upRow = upBase + m * inputDimension;

            float gate = 0.0f;
            float up = 0.0f;
            for (uint j = tiisg + split * SIMD_WIDTH; j < inputDimension; j += splitCount * SIMD_WIDTH) {
                const float x = float(inputRow[j]);
                gate += \(readWeight("gateRow[j]")) * x;
                up += \(readWeight("upRow[j]")) * x;
            }
            gate = simd_sum(gate);
            up = simd_sum(up);
            const uint partialIndex = localRow * splitCount + split;
            threadgroup float* gatePartials = partials;
            threadgroup float* upPartials = partials + rowsPerThreadgroup * splitCount;
            if (tiisg == 0) {
                gatePartials[partialIndex] = gate;
                upPartials[partialIndex] = up;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (split == 0u && tiisg == 0) {
                const uint base = localRow * splitCount;
                const float fullGate = gatePartials[base] + gatePartials[base + 1u];
                const float fullUp = upPartials[base] + upPartials[base + 1u];
                activationScratch[k * intermediateDimension + m] =
                    fullGate * (1.0f / (1.0f + exp(-fullGate))) * fullUp * routeWeight;
            }
        }
        """
    }

    private static func generateSparseMoEGateUpPacked4(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard weightFormat.isBFloat16 else {
            return generateSparseMoEGateUp(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }

        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType

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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint seqPos = gid.y;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint flat = gid.x * simdgroupsPerThreadgroup + sgitg;
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
            device float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            const uint expert = uint(selectedExpertScratch[k]);
            const float routeWeight = selectedWeightScratch[k];
            device const \(wt)* gateBase = expertGateUpWeight
                + expert * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase = gateBase + intermediateDimension * inputDimension;
            device const \(wt)* gateRow = gateBase + m * inputDimension;
            device const \(wt)* upRow = upBase + m * inputDimension;

            float gate = 0.0f;
            float up = 0.0f;
            uint j = tiisg * 4u;
            device const \(bt)4* inputLane = (device const \(bt)4*)inputRow + tiisg;
            device const ushort4* gateLane = (device const ushort4*)gateRow + tiisg;
            device const ushort4* upLane = (device const ushort4*)upRow + tiisg;
            for (; j + 3u < inputDimension; j += SIMD_WIDTH * 4u) {
                const float4 x = float4(inputLane[0]);
                const float4 gateWeight = bf16x4_to_float4(gateLane[0]);
                const float4 upWeight = bf16x4_to_float4(upLane[0]);
                gate += dot(gateWeight, x);
                up += dot(upWeight, x);
                inputLane += SIMD_WIDTH;
                gateLane += SIMD_WIDTH;
                upLane += SIMD_WIDTH;
            }
            for (; j < inputDimension; j++) {
                const float x = float(inputRow[j]);
                gate += bf16_to_float(gateRow[j]) * x;
                up += bf16_to_float(upRow[j]) * x;
            }
            gate = simd_sum(gate);
            up = simd_sum(up);
            if (tiisg == 0) {
                activationScratch[k * intermediateDimension + m] =
                    gate * (1.0f / (1.0f + exp(-gate))) * up * routeWeight;
            }
        }
        """
    }

    private static func generateSparseMoEGateUpPacked8(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard weightFormat.isBFloat16 else {
            return generateSparseMoEGateUp(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }

        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType

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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint seqPos = gid.y;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint flat = gid.x * simdgroupsPerThreadgroup + sgitg;
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
            device float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            const uint expert = uint(selectedExpertScratch[k]);
            const float routeWeight = selectedWeightScratch[k];
            device const \(wt)* gateBase = expertGateUpWeight
                + expert * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase = gateBase + intermediateDimension * inputDimension;
            device const \(wt)* gateRow = gateBase + m * inputDimension;
            device const \(wt)* upRow = upBase + m * inputDimension;

            float gate = 0.0f;
            float up = 0.0f;
            uint j = tiisg * 8u;
            device const \(bt)4* inputLane = (device const \(bt)4*)inputRow + tiisg * 2u;
            device const ushort4* gateLane = (device const ushort4*)gateRow + tiisg * 2u;
            device const ushort4* upLane = (device const ushort4*)upRow + tiisg * 2u;
            for (; j + 7u < inputDimension; j += SIMD_WIDTH * 8u) {
                const float4 x0 = float4(inputLane[0]);
                const float4 x1 = float4(inputLane[1]);
                const float4 gateWeight0 = bf16x4_to_float4(gateLane[0]);
                const float4 gateWeight1 = bf16x4_to_float4(gateLane[1]);
                const float4 upWeight0 = bf16x4_to_float4(upLane[0]);
                const float4 upWeight1 = bf16x4_to_float4(upLane[1]);
                gate += dot(gateWeight0, x0) + dot(gateWeight1, x1);
                up += dot(upWeight0, x0) + dot(upWeight1, x1);
                inputLane += SIMD_WIDTH * 2u;
                gateLane += SIMD_WIDTH * 2u;
                upLane += SIMD_WIDTH * 2u;
            }
            for (; j < inputDimension; j++) {
                const float x = float(inputRow[j]);
                gate += bf16_to_float(gateRow[j]) * x;
                up += bf16_to_float(upRow[j]) * x;
            }
            gate = simd_sum(gate);
            up = simd_sum(up);
            if (tiisg == 0) {
                activationScratch[k * intermediateDimension + m] =
                    gate * (1.0f / (1.0f + exp(-gate))) * up * routeWeight;
            }
        }
        """
    }

    private static func generateSparseMoEGateUpRow2Packed4(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard weightFormat.isBFloat16 else {
            return generateSparseMoEGateUp(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }

        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType

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
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint seqPos = gid.y;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            const uint baseFlat = (gid.x * simdgroupsPerThreadgroup + sgitg) * 2u;
            const uint totalRows = expertsPerToken * intermediateDimension;
            if (seqPos >= sequenceLength || baseFlat >= totalRows) {
                return;
            }

            const uint k0 = baseFlat / intermediateDimension;
            const uint m0 = baseFlat - k0 * intermediateDimension;
            const uint flat1 = baseFlat + 1u;
            const bool hasRow1 = flat1 < totalRows;
            const uint k1 = hasRow1 ? flat1 / intermediateDimension : k0;
            const uint m1 = hasRow1 ? flat1 - k1 * intermediateDimension : m0;

            device const \(bt)* inputRow = input + seqPos * inputRowStride;
            device float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device float* selectedExpertScratch = scratchRow;
            device float* selectedWeightScratch = scratchRow + expertsPerToken;
            device float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            const uint expert0 = uint(selectedExpertScratch[k0]);
            const float routeWeight0 = selectedWeightScratch[k0];
            device const \(wt)* gateBase0 = expertGateUpWeight
                + expert0 * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase0 = gateBase0 + intermediateDimension * inputDimension;
            device const \(wt)* gateRow0 = gateBase0 + m0 * inputDimension;
            device const \(wt)* upRow0 = upBase0 + m0 * inputDimension;

            const uint expert1 = hasRow1 ? uint(selectedExpertScratch[k1]) : expert0;
            const float routeWeight1 = hasRow1 ? selectedWeightScratch[k1] : routeWeight0;
            device const \(wt)* gateBase1 = expertGateUpWeight
                + expert1 * (2u * intermediateDimension * inputDimension);
            device const \(wt)* upBase1 = gateBase1 + intermediateDimension * inputDimension;
            device const \(wt)* gateRow1 = gateBase1 + m1 * inputDimension;
            device const \(wt)* upRow1 = upBase1 + m1 * inputDimension;

            float gate0 = 0.0f;
            float up0 = 0.0f;
            float gate1 = 0.0f;
            float up1 = 0.0f;
            uint j = tiisg * 4u;
            device const \(bt)4* inputLane = (device const \(bt)4*)inputRow + tiisg;
            device const ushort4* gateLane0 = (device const ushort4*)gateRow0 + tiisg;
            device const ushort4* upLane0 = (device const ushort4*)upRow0 + tiisg;
            device const ushort4* gateLane1 = (device const ushort4*)gateRow1 + tiisg;
            device const ushort4* upLane1 = (device const ushort4*)upRow1 + tiisg;
            for (; j + 3u < inputDimension; j += SIMD_WIDTH * 4u) {
                const float4 x = float4(inputLane[0]);
                gate0 += dot(bf16x4_to_float4(gateLane0[0]), x);
                up0 += dot(bf16x4_to_float4(upLane0[0]), x);
                if (hasRow1) {
                    gate1 += dot(bf16x4_to_float4(gateLane1[0]), x);
                    up1 += dot(bf16x4_to_float4(upLane1[0]), x);
                }
                inputLane += SIMD_WIDTH;
                gateLane0 += SIMD_WIDTH;
                upLane0 += SIMD_WIDTH;
                gateLane1 += SIMD_WIDTH;
                upLane1 += SIMD_WIDTH;
            }
            for (; j < inputDimension; j++) {
                const float x = float(inputRow[j]);
                gate0 += bf16_to_float(gateRow0[j]) * x;
                up0 += bf16_to_float(upRow0[j]) * x;
                if (hasRow1) {
                    gate1 += bf16_to_float(gateRow1[j]) * x;
                    up1 += bf16_to_float(upRow1[j]) * x;
                }
            }
            gate0 = simd_sum(gate0);
            up0 = simd_sum(up0);
            gate1 = simd_sum(gate1);
            up1 = simd_sum(up1);
            if (tiisg == 0) {
                activationScratch[k0 * intermediateDimension + m0] =
                    gate0 * (1.0f / (1.0f + exp(-gate0))) * up0 * routeWeight0;
                if (hasRow1) {
                    activationScratch[k1 * intermediateDimension + m1] =
                        gate1 * (1.0f / (1.0f + exp(-gate1))) * up1 * routeWeight1;
                }
            }
        }
        """
    }

    private static func generateSparseMoEDownSplit2(
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
            threadgroup float* partials            [[threadgroup(0)]],
            uint2 gid                              [[threadgroup_position_in_grid]],
            uint tiisg                             [[thread_index_in_simdgroup]],
            uint sgitg                             [[simdgroup_index_in_threadgroup]],
            uint2 threadsPerThreadgroup            [[threads_per_threadgroup]]
        ) {
            const uint splitCount = 2u;
            const uint simdgroupsPerThreadgroup = max(1u, threadsPerThreadgroup.x / SIMD_WIDTH);
            if (simdgroupsPerThreadgroup < splitCount) {
                return;
            }
            const uint rowsPerThreadgroup = max(1u, simdgroupsPerThreadgroup / splitCount);
            const uint localRow = sgitg / splitCount;
            const uint split = sgitg - localRow * splitCount;
            const uint row = gid.x * rowsPerThreadgroup + localRow;
            const uint seqPos = gid.y;
            if (seqPos >= sequenceLength || row >= outputDimension) {
                return;
            }

            device const float* scratchRow = moeScratch + seqPos * scratchRowStride;
            device const float* selectedExpertScratch = scratchRow;
            device const float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            float total = 0.0f;
            for (uint k = 0; k < expertsPerToken; k++) {
                const uint expert = uint(selectedExpertScratch[k]);
                device const \(wt)* downRow = expertDownWeight
                    + expert * (outputDimension * intermediateDimension)
                    + row * intermediateDimension;
                float partial = 0.0f;
                for (uint m = tiisg + split * SIMD_WIDTH; m < intermediateDimension; m += splitCount * SIMD_WIDTH) {
                    partial += \(readWeight("downRow[m]")) *
                        activationScratch[k * intermediateDimension + m];
                }
                total += simd_sum(partial);
            }

            const uint partialIndex = localRow * splitCount + split;
            if (tiisg == 0) {
                partials[partialIndex] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (split == 0u && tiisg == 0) {
                const uint base = localRow * splitCount;
                total = partials[base] + partials[base + 1u];
                output[seqPos * outputRowStride + row] = \(bt)(\(storedOutput));
            }
        }
        """
    }

    private static func generateSparseMoEDownPacked4(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard weightFormat.isBFloat16 else {
            return generateSparseMoEDown(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }

        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
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
            device const float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            float total = 0.0f;
            for (uint k = 0; k < expertsPerToken; k++) {
                const uint expert = uint(selectedExpertScratch[k]);
                device const \(wt)* downRow = expertDownWeight
                    + expert * (outputDimension * intermediateDimension)
                    + row * intermediateDimension;
                device const float* activatedRow = activationScratch + k * intermediateDimension;
                float partial = 0.0f;
                uint m = tiisg * 4u;
                device const ushort4* downLane = (device const ushort4*)downRow + tiisg;
                device const float4* activatedLane = (device const float4*)activatedRow + tiisg;
                for (; m + 3u < intermediateDimension; m += SIMD_WIDTH * 4u) {
                    const float4 weight = bf16x4_to_float4(downLane[0]);
                    partial += dot(weight, activatedLane[0]);
                    downLane += SIMD_WIDTH;
                    activatedLane += SIMD_WIDTH;
                }
                for (; m < intermediateDimension; m++) {
                    partial += bf16_to_float(downRow[m]) * activatedRow[m];
                }
                total += simd_sum(partial);
            }
            if (tiisg == 0) {
                output[seqPos * outputRowStride + row] = \(bt)(\(storedOutput));
            }
        }
        """
    }

    private static func generateSparseMoEDownPacked8(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        guard weightFormat.isBFloat16 else {
            return generateSparseMoEDown(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat
            )
        }

        let bt = bufferPrecision.metalType
        let wt = weightFormat.bufferType
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
            device const float* activationScratch = scratchRow + 2u * expertsPerToken + 2u * 128u;

            float total = 0.0f;
            for (uint k = 0; k < expertsPerToken; k++) {
                const uint expert = uint(selectedExpertScratch[k]);
                device const \(wt)* downRow = expertDownWeight
                    + expert * (outputDimension * intermediateDimension)
                    + row * intermediateDimension;
                device const float* activatedRow = activationScratch + k * intermediateDimension;
                float partial = 0.0f;
                uint m = tiisg * 8u;
                device const ushort4* downLane = (device const ushort4*)downRow + tiisg * 2u;
                device const float4* activatedLane = (device const float4*)activatedRow + tiisg * 2u;
                for (; m + 7u < intermediateDimension; m += SIMD_WIDTH * 8u) {
                    const float4 weight0 = bf16x4_to_float4(downLane[0]);
                    const float4 weight1 = bf16x4_to_float4(downLane[1]);
                    partial += dot(weight0, activatedLane[0]) + dot(weight1, activatedLane[1]);
                    downLane += SIMD_WIDTH * 2u;
                    activatedLane += SIMD_WIDTH * 2u;
                }
                for (; m < intermediateDimension; m++) {
                    partial += bf16_to_float(downRow[m]) * activatedRow[m];
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
