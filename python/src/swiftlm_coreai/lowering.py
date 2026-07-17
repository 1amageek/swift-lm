"""Generic lowering from Swift LMIR documents to exportable PyTorch graphs."""

from __future__ import annotations

from typing import Any

from .document import ExportDocument
from .errors import ExportError
from .weights import SafetensorWeightStore


class TorchGraphLowerer:
    """Compile a Swift-authored graph without model-family Python code."""

    def __init__(
        self,
        document: ExportDocument,
        weights: SafetensorWeightStore,
        torch: Any,
    ) -> None:
        self.document = document
        self.weights = weights
        self.torch = torch
        _validate_lowering_contract(document.raw["rootRegion"], "root")

    def make_stateless_model(self) -> Any:
        region = _RegionModule(
            self.document.raw["rootRegion"],
            self.weights,
            self.torch,
        )
        torch = self.torch

        class LoweredModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = region

            def forward(self, input_ids: Any, position_ids: Any) -> Any:
                outputs = self.region((), input_ids, position_ids, {})
                if len(outputs) != 1:
                    raise RuntimeError("Language-model graphs must produce one logits tensor")
                return outputs[0]

        return LoweredModel()

    def make_stateful_model(self) -> Any:
        state_names = tuple(state["name"] for state in self.document.function["states"])
        region = _RegionModule(
            self.document.raw["rootRegion"],
            self.weights,
            self.torch,
        )
        torch = self.torch

        class LoweredModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.region = region
                self.state_names = state_names

            def forward(
                self,
                input_ids: Any,
                position_ids: Any,
                states: tuple[Any, ...],
            ) -> Any:
                if len(states) != len(self.state_names):
                    raise RuntimeError("State tensor count does not match the Swift contract")
                state_values = {
                    name: value
                    for name, value in zip(self.state_names, states, strict=True)
                }
                outputs = self.region((), input_ids, position_ids, state_values)
                if len(outputs) != 1:
                    raise RuntimeError("Language-model graphs must produce one logits tensor")
                return outputs[0]

        return LoweredModel()

    def make_model(self) -> Any:
        if self.document.execution == "stateless":
            return self.make_stateless_model()
        if self.document.execution == "stateful":
            return self.make_stateful_model()
        raise ExportError("unsupported_execution", self.document.execution)


class _RegionModule:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = args[2]

        class Region(torch.nn.Module):
            def __init__(self, payload: dict[str, Any], weights: SafetensorWeightStore) -> None:
                super().__init__()
                self.parameter_ids = tuple(payload["parameters"])
                self.result_ids = tuple(payload["results"])
                self.operations = torch.nn.ModuleList(
                    _make_operation(operation, weights, torch)
                    for operation in payload["operations"]
                )

            def forward(
                self,
                parameters: tuple[Any, ...],
                input_ids: Any,
                position_ids: Any,
                states: dict[str, Any],
                iteration_index: int | None = None,
            ) -> tuple[Any, ...]:
                values = {
                    value_id: value
                    for value_id, value in zip(self.parameter_ids, parameters, strict=True)
                }
                for operation in self.operations:
                    operands = tuple(values[value_id] for value_id in operation.operand_ids)
                    results = operation(
                        operands,
                        input_ids,
                        position_ids,
                        states,
                        iteration_index,
                    )
                    for value_id, value in zip(operation.result_ids, results, strict=True):
                        values[value_id] = value
                return tuple(values[value_id] for value_id in self.result_ids)

        return Region(args[0], args[1])


def _make_operation(
    payload: dict[str, Any],
    weights: SafetensorWeightStore,
    torch: Any,
) -> Any:
    kind = payload["kind"]
    tag = kind["tag"]
    if tag == "primitive":
        implementation = _make_primitive(
            kind["primitive"],
            payload["parameterBindings"],
            payload["stateBindings"],
            weights,
            torch,
        )

        class PrimitiveOperation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.implementation = implementation
                self.operand_ids = tuple(payload["operands"])
                self.result_ids = tuple(payload["results"])

            def forward(
                self,
                operands: tuple[Any, ...],
                input_ids: Any,
                position_ids: Any,
                states: dict[str, Any],
                iteration_index: int | None,
            ) -> tuple[Any, ...]:
                result = self.implementation(operands, input_ids, position_ids, states)
                return (result,)

        return PrimitiveOperation()

    if tag == "residual":
        strategy = _enum_case(kind["strategy"])
        if strategy != "add":
            raise ExportError("unsupported_residual_strategy", strategy)
        body = _RegionModule(kind["body"], weights, torch)

        class ResidualOperation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = body
                self.operand_ids = tuple(payload["operands"])
                self.result_ids = tuple(payload["results"])

            def forward(
                self,
                operands: tuple[Any, ...],
                input_ids: Any,
                position_ids: Any,
                states: dict[str, Any],
                iteration_index: int | None,
            ) -> tuple[Any, ...]:
                body_outputs = self.body(
                    operands,
                    input_ids,
                    position_ids,
                    states,
                    iteration_index,
                )
                return tuple(
                    operand + body_output
                    for operand, body_output in zip(operands, body_outputs, strict=True)
                )

        return ResidualOperation()

    if tag == "parallel":
        merge = _enum_case(kind["merge"])
        branches = torch.nn.ModuleList(
            _RegionModule(branch, weights, torch)
            for branch in kind["branches"]
        )

        class ParallelOperation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branches = branches
                self.operand_ids = tuple(payload["operands"])
                self.result_ids = tuple(payload["results"])

            def forward(
                self,
                operands: tuple[Any, ...],
                input_ids: Any,
                position_ids: Any,
                states: dict[str, Any],
                iteration_index: int | None,
            ) -> tuple[Any, ...]:
                branch_outputs = [
                    branch(
                        operands,
                        input_ids,
                        position_ids,
                        states,
                        iteration_index,
                    )
                    for branch in self.branches
                ]
                if merge == "add":
                    return tuple(
                        sum(outputs)
                        for outputs in zip(*branch_outputs, strict=True)
                    )
                raise RuntimeError(f"Unsupported parallel merge {merge}")

        return ParallelOperation()

    if tag == "repeating":
        count = kind["count"]
        body = _RegionModule(kind["body"], weights, torch)

        class RepeatingOperation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = body
                self.operand_ids = tuple(payload["operands"])
                self.result_ids = tuple(payload["results"])

            def forward(
                self,
                operands: tuple[Any, ...],
                input_ids: Any,
                position_ids: Any,
                states: dict[str, Any],
                iteration_index: int | None,
            ) -> tuple[Any, ...]:
                values = operands
                for index in range(count):
                    values = self.body(
                        values,
                        input_ids,
                        position_ids,
                        states,
                        index,
                    )
                return values

        return RepeatingOperation()

    if tag == "conditional":
        selected_indices = _layer_indices(kind["condition"])
        then_body = _RegionModule(kind["then"], weights, torch)
        else_body = _RegionModule(kind["else"], weights, torch)

        class ConditionalOperation(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.then_body = then_body
                self.else_body = else_body
                self.operand_ids = tuple(payload["operands"])
                self.result_ids = tuple(payload["results"])

            def forward(
                self,
                operands: tuple[Any, ...],
                input_ids: Any,
                position_ids: Any,
                states: dict[str, Any],
                iteration_index: int | None,
            ) -> tuple[Any, ...]:
                if iteration_index is None:
                    raise RuntimeError("Layer-index condition requires a repeating parent")
                body = (
                    self.then_body
                    if iteration_index in selected_indices
                    else self.else_body
                )
                return body(
                    operands,
                    input_ids,
                    position_ids,
                    states,
                    iteration_index,
                )

        return ConditionalOperation()

    raise ExportError("unsupported_operation", tag)


def _make_primitive(
    primitive: dict[str, Any],
    bindings: list[dict[str, str]],
    state_bindings: list[dict[str, Any]],
    weights: SafetensorWeightStore,
    torch: Any,
) -> Any:
    opcode = primitive["opcode"]
    attributes = primitive["attributes"]
    tensors = {}
    for binding in bindings:
        role = binding["role"]
        dtype = None
        if opcode == "state_space" and role in {"A_log", "dt_bias", "scale"}:
            dtype = torch.float32
        tensors[role] = weights.tensor(binding["tensorName"], dtype=dtype)
    states_by_role = {binding["role"]: binding for binding in state_bindings}

    class Primitive(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.opcode = opcode
            self.attributes = attributes
            self.states_by_role = states_by_role
            for role, tensor in tensors.items():
                if (
                    opcode == "moe"
                    and role.startswith("expert_")
                    and role != "expert_bias"
                ):
                    continue
                self.register_parameter(
                    _parameter_name(role),
                    torch.nn.Parameter(tensor, requires_grad=False),
                )
            if opcode == "attention":
                from coreai_models.primitives.macos.sdpa import SDPA

                scale = attributes.get("attentionScale")
                if scale is None:
                    scale = attributes["headDimension"] ** -0.5
                self.sdpa = SDPA(scale=scale, is_causal=attributes["causal"])
            if opcode == "moe":
                from coreai_models.primitives.macos.switch import SwitchGLU

                expert_mlp = attributes["expertMLP"]
                self.switch_glu = SwitchGLU(
                    expert_mlp["inputSize"],
                    expert_mlp["intermediateSize"],
                    attributes["expertCount"],
                    bias=False,
                )
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    stacked = torch.stack(
                        [
                            tensors[f"expert_{index}_{projection}"]
                            for index in range(attributes["expertCount"])
                        ],
                        dim=0,
                    ).unsqueeze(0)
                    switch_projection = getattr(self.switch_glu, projection)
                    switch_projection.weight = torch.nn.Parameter(
                        stacked,
                        requires_grad=False,
                    )
            if opcode == "state_space":
                from coreai_torch.composite_ops import GatedDeltaUpdate

                self.gated_delta_update = GatedDeltaUpdate(use_qk_l2_norm=True)

        def forward(
            self,
            operands: tuple[Any, ...],
            input_ids: Any,
            position_ids: Any,
            states: dict[str, Any],
        ) -> Any:
            if self.opcode == "token_embedding":
                output = torch.nn.functional.embedding(input_ids, self.embedding_table)
                scale = self.attributes.get("embeddingScale")
                return output if scale is None else output * scale
            if len(operands) != 1:
                raise RuntimeError(f"{self.opcode} requires one operand")
            value = operands[0]
            if self.opcode == "rms_norm":
                return self._rms_norm(value)
            if self.opcode == "layer_norm":
                return self._layer_norm(value)
            if self.opcode == "layer_scale":
                return value * self.layer_scalar
            if self.opcode == "linear":
                return torch.nn.functional.linear(
                    value,
                    self.weight,
                    getattr(self, "bias", None),
                )
            if self.opcode == "mlp":
                return self._mlp(value)
            if self.opcode == "moe":
                return self._moe(value)
            if self.opcode == "state_space":
                return self._state_space(value, states)
            if self.opcode == "short_conv":
                return self._short_conv(value, states)
            if self.opcode == "attention":
                return self._attention(value, position_ids, states)
            if self.opcode == "output_head":
                bias = getattr(self, "bias", None)
                return torch.nn.functional.linear(value, self.weight, bias)
            raise RuntimeError(f"Unsupported primitive {self.opcode}")

        def _rms_norm(self, value: Any) -> Any:
            input_dtype = value.dtype
            normalized = value.float()
            variance = normalized.pow(2).mean(-1, keepdim=True)
            normalized = normalized * torch.rsqrt(variance + self.attributes["epsilon"])
            normalized = normalized.to(input_dtype)
            if not self.attributes.get("withScale", True):
                return normalized
            scale = self.scale + self.attributes.get("weightBias", 0)
            return normalized * scale

        def _layer_norm(self, value: Any) -> Any:
            weight = self.scale if self.attributes["affine"] else None
            bias = getattr(self, "bias", None) if self.attributes["affine"] else None
            return torch.nn.functional.layer_norm(
                value,
                (self.attributes["dimension"],),
                weight,
                bias,
                self.attributes["epsilon"],
            )

        def _mlp(self, value: Any) -> Any:
            activation = _enum_case(self.attributes["activation"])
            gating = _enum_case(self.attributes["gating"])
            up = torch.nn.functional.linear(
                value,
                self.up_proj,
                getattr(self, "up_proj_bias", None),
            )
            if gating == "none":
                hidden = _activate(up, activation, torch)
            else:
                gate = torch.nn.functional.linear(
                    value,
                    self.gate_proj,
                    getattr(self, "gate_proj_bias", None),
                )
                gate_activation = {
                    "glu": "sigmoid",
                    "geglu": "gelu",
                    "swiglu": "silu",
                }[gating]
                hidden = _activate(gate, gate_activation, torch) * up
            return torch.nn.functional.linear(
                hidden,
                self.down_proj,
                getattr(self, "down_proj_bias", None),
            )

        def _moe(self, value: Any) -> Any:
            router_logits = torch.nn.functional.linear(value, self.router).float()
            gate_kind = _enum_case(self.attributes["gateKind"])
            top_k = self.attributes["expertsPerToken"]
            normalize = self.attributes["normalizeRoutingWeights"]

            if gate_kind == "topK":
                if normalize:
                    top_logits, active_indices = torch.topk(
                        router_logits,
                        top_k,
                        dim=-1,
                        largest=True,
                    )
                    active_scores = torch.softmax(top_logits, dim=-1)
                else:
                    routing_scores = torch.softmax(router_logits, dim=-1)
                    active_scores, active_indices = torch.topk(
                        routing_scores,
                        top_k,
                        dim=-1,
                        largest=True,
                    )
            elif gate_kind == "sigmoidTopK":
                routing_scores = torch.sigmoid(router_logits)
                selection_scores = routing_scores
                if self.attributes["useExpertBias"]:
                    selection_scores = selection_scores + self.expert_bias.float()
                _, active_indices = torch.topk(
                    selection_scores,
                    top_k,
                    dim=-1,
                    largest=True,
                )
                active_scores = torch.gather(routing_scores, -1, active_indices)
                if normalize:
                    active_scores = active_scores / active_scores.sum(
                        dim=-1,
                        keepdim=True,
                    )
            else:
                raise RuntimeError(f"Unsupported MoE gate kind {gate_kind}")

            active_scores = active_scores * self.attributes["routedScalingFactor"]
            active_outputs = self.switch_glu(value, active_indices.to(torch.uint16))
            weighted = active_outputs * active_scores.to(value.dtype).unsqueeze(-1)
            return weighted.sum(dim=-2).to(value.dtype)

        def _state_space(self, value: Any, states: dict[str, Any]) -> Any:
            batch_size, sequence_length, _ = value.shape
            head_count = self.attributes["numHeads"]
            group_count = self.attributes["groupCount"]
            key_head_dimension = self.attributes["keyHeadDim"]
            value_head_dimension = self.attributes["valueHeadDim"]
            key_dimension = group_count * key_head_dimension
            value_dimension = head_count * value_head_dimension
            convolution_dimension = 2 * key_dimension + value_dimension
            kernel_size = self.attributes["convKernelSize"]

            mixed_qkv = torch.nn.functional.linear(value, self.in_proj_qkv)
            mixed_qkv = mixed_qkv.transpose(-1, -2)
            if "conv_cache" in self.states_by_role:
                conv_cache, conv_index = self._state("conv_cache", states)
                previous = conv_cache.narrow(0, conv_index, 1).squeeze(0)
                padded_qkv = torch.cat((previous, mixed_qkv), dim=-1)
            else:
                padded_qkv = torch.nn.functional.pad(
                    mixed_qkv,
                    (kernel_size - 1, 0),
                )
            mixed_qkv = torch.nn.functional.conv1d(
                padded_qkv,
                self.conv_weight,
                groups=convolution_dimension,
            )[..., -sequence_length:]
            mixed_qkv = torch.nn.functional.silu(mixed_qkv).transpose(-1, -2)

            if "conv_cache" in self.states_by_role:
                updated_conv = padded_qkv[..., -kernel_size:]
                self._update_state(
                    conv_cache,
                    conv_index,
                    updated_conv,
                    value.device,
                )

            query, key, projected_value = torch.split(
                mixed_qkv,
                (key_dimension, key_dimension, value_dimension),
                dim=-1,
            )
            query = query.reshape(
                batch_size,
                sequence_length,
                group_count,
                key_head_dimension,
            )
            key = key.reshape(
                batch_size,
                sequence_length,
                group_count,
                key_head_dimension,
            )
            projected_value = projected_value.reshape(
                batch_size,
                sequence_length,
                head_count,
                value_head_dimension,
            )
            repetitions = head_count // group_count
            query = query.repeat_interleave(repetitions, dim=2)
            key = key.repeat_interleave(repetitions, dim=2)

            beta = torch.sigmoid(torch.nn.functional.linear(value, self.in_proj_b))
            decay_input = torch.nn.functional.linear(value, self.in_proj_a)
            decay = -self.A_log.float().exp() * torch.nn.functional.softplus(
                decay_input.float() + self.dt_bias.float()
            )
            gate = torch.nn.functional.linear(value, self.in_proj_z).reshape(
                batch_size,
                sequence_length,
                head_count,
                value_head_dimension,
            )

            if "recurrent_state" in self.states_by_role:
                recurrent_cache, recurrent_index = self._state(
                    "recurrent_state",
                    states,
                )
                initial_state = recurrent_cache.narrow(
                    0,
                    recurrent_index,
                    1,
                ).squeeze(0)
            else:
                initial_state = torch.zeros(
                    batch_size,
                    head_count,
                    key_head_dimension,
                    value_head_dimension,
                    dtype=torch.float32,
                    device=value.device,
                )

            recurrence_output, updated_recurrent = self.gated_delta_update(
                query.transpose(1, 2).float(),
                key.transpose(1, 2).float(),
                projected_value.transpose(1, 2).float(),
                decay.transpose(1, 2).float(),
                beta.transpose(1, 2).float(),
                initial_state,
            )
            if "recurrent_state" in self.states_by_role:
                self._update_state(
                    recurrent_cache,
                    recurrent_index,
                    updated_recurrent,
                    value.device,
                )

            recurrence_output = recurrence_output.to(value.dtype)
            normalized = recurrence_output.float()
            variance = normalized.pow(2).mean(-1, keepdim=True)
            normalized = normalized * torch.rsqrt(
                variance + self.attributes["normEpsilon"]
            )
            normalized = normalized * self.scale.float()
            normalized = normalized * torch.nn.functional.silu(gate.float())
            normalized = normalized.to(value.dtype).reshape(
                batch_size,
                sequence_length,
                value_dimension,
            )
            return torch.nn.functional.linear(normalized, self.out_proj)

        def _update_state(
            self,
            cache: Any,
            axis_index: int,
            update: Any,
            device: Any,
        ) -> None:
            from coreai_models.primitives._ops import mutable_slice_update

            begin = torch.tensor(
                (axis_index,) + (0,) * (cache.dim() - 1),
                dtype=torch.int32,
                device=device,
            )
            end = torch.tensor(
                (axis_index + 1,) + tuple(cache.shape[1:]),
                dtype=torch.int32,
                device=device,
            )
            mutable_slice_update(
                x=cache,
                update=update.unsqueeze(0),
                begin=begin,
                end=end,
            )

        def _short_conv(self, value: Any, states: dict[str, Any]) -> Any:
            projected = torch.nn.functional.linear(
                value,
                self.in_proj,
                getattr(self, "in_proj_bias", None),
            ).transpose(-1, -2)
            gate, candidate, source = projected.chunk(3, dim=-2)
            mixed = gate * source
            kernel_size = self.attributes["kernelSize"]
            if "conv_cache" in self.states_by_role:
                cache, axis_index = self._state("conv_cache", states)
                previous = cache.narrow(0, axis_index, 1).squeeze(0)
                padded = torch.cat((previous, mixed), dim=-1)
            else:
                padded = torch.nn.functional.pad(mixed, (kernel_size - 1, 0))
            convolution = torch.nn.functional.conv1d(
                padded,
                self.conv_weight,
                getattr(self, "conv_bias", None),
                groups=self.attributes["hiddenSize"],
            )
            if "conv_cache" in self.states_by_role:
                from coreai_models.primitives._ops import mutable_slice_update

                convolution = convolution[..., -mixed.shape[-1] :]
                updated = padded[..., -kernel_size:]
                begin = torch.tensor(
                    (axis_index, 0, 0, 0),
                    dtype=torch.int32,
                    device=value.device,
                )
                end = torch.tensor(
                    (axis_index + 1, cache.size(1), cache.size(2), cache.size(3)),
                    dtype=torch.int32,
                    device=value.device,
                )
                mutable_slice_update(
                    x=cache,
                    update=updated.unsqueeze(0),
                    begin=begin,
                    end=end,
                )
            output = candidate * convolution
            return torch.nn.functional.linear(
                output.transpose(-1, -2).contiguous(),
                self.out_proj,
                getattr(self, "out_proj_bias", None),
            )

        def _attention(
            self,
            value: Any,
            position_ids: Any,
            states: dict[str, Any],
        ) -> Any:
            head_count = self.attributes["headCount"]
            kv_head_count = self.attributes["kvHeadCount"]
            head_dimension = self.attributes["headDimension"]
            input_shape = value.shape[:-1]
            query = torch.nn.functional.linear(
                value,
                self.q_proj,
                getattr(self, "q_proj_bias", None),
            )
            output_gate = self.attributes.get("outputGate")
            gate = None
            if output_gate is not None:
                output_gate_kind = _enum_case(output_gate)
                if output_gate_kind == "sigmoidPackedInQProj":
                    query = query.view(
                        *input_shape,
                        head_count,
                        head_dimension * 2,
                    )
                    query, gate = query.chunk(2, dim=-1)
                    gate = gate.reshape(*input_shape, -1)
                else:
                    raise RuntimeError(
                        f"Unsupported attention output gate {output_gate_kind}"
                    )
            key = torch.nn.functional.linear(
                value,
                self.k_proj,
                getattr(self, "k_proj_bias", None),
            )
            projected_value = torch.nn.functional.linear(
                value,
                self.v_proj,
                getattr(self, "v_proj_bias", None),
            )
            if gate is None:
                query = query.view(*input_shape, head_count, head_dimension)
            key = key.view(*input_shape, kv_head_count, head_dimension)
            projected_value = projected_value.view(
                *input_shape, kv_head_count, head_dimension
            )
            query = self._qk_norm(query, "q_layernorm").transpose(1, 2)
            key = self._qk_norm(key, "k_layernorm").transpose(1, 2)
            projected_value = projected_value.transpose(1, 2)
            query_positions = position_ids
            if "key_cache" in self.states_by_role:
                query_length = query.shape[-2]
                offset = position_ids.shape[-1] - query_length
                query_positions = position_ids.narrow(-1, offset, query_length)
            query, key = self._apply_rope(query, key, query_positions)
            if "key_cache" in self.states_by_role:
                from coreai_models.primitives.macos.cache import KVCache

                key_cache, key_index = self._state("key_cache", states)
                value_cache, value_index = self._state("value_cache", states)
                if key_index != value_index:
                    raise RuntimeError("Key and value cache axis indices must match")
                cache = KVCache(key_cache, value_cache)
                key, projected_value = cache.update_and_fetch(
                    key_index,
                    offset,
                    key,
                    projected_value,
                    seq_len=position_ids.shape[-1],
                    query_len=query_length,
                )
            repetitions = head_count // kv_head_count
            key = _repeat_kv(key, repetitions)
            projected_value = _repeat_kv(projected_value, repetitions)
            output = self.sdpa(query, key, projected_value)
            output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            if gate is not None:
                output = output * torch.sigmoid(gate)
            return torch.nn.functional.linear(
                output,
                self.o_proj,
                getattr(self, "o_proj_bias", None),
            )

        def _state(
            self,
            role: str,
            states: dict[str, Any],
        ) -> tuple[Any, int]:
            binding = self.states_by_role[role]
            return states[binding["state"]], binding["axisIndex"]

        def _qk_norm(self, value: Any, parameter: str) -> Any:
            kind = self.attributes.get("qkNorm")
            if kind is None or _enum_case(kind) == "none":
                return value
            if _enum_case(kind) not in {"rmsNorm", "rmsNormUnitOffset"}:
                raise RuntimeError(f"Unsupported QK norm: {_enum_case(kind)}")
            weight = getattr(self, parameter)
            input_dtype = value.dtype
            normalized = value.float()
            variance = normalized.pow(2).mean(-1, keepdim=True)
            normalized = normalized * torch.rsqrt(
                variance + self.attributes.get("qkNormEpsilon", 1e-6)
            )
            weight_bias = 1 if _enum_case(kind) == "rmsNormUnitOffset" else 0
            return normalized.to(input_dtype) * (weight + weight_bias)

        def _apply_rope(self, query: Any, key: Any, position_ids: Any) -> tuple[Any, Any]:
            rope = self.attributes.get("rope")
            if rope is None:
                return query, key
            dimension = rope["dimension"]
            base = rope["base"]
            indices = torch.arange(0, dimension, 2, device=query.device).float()
            inverse_frequency = 1.0 / (base ** (indices / dimension))
            frequencies = position_ids.float().unsqueeze(-1) * inverse_frequency
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine = embedding.cos().to(query.dtype).unsqueeze(1)
            sine = embedding.sin().to(query.dtype).unsqueeze(1)
            query_rotary = query[..., :dimension]
            key_rotary = key[..., :dimension]
            query_rotary = (query_rotary * cosine) + (_rotate_half(query_rotary, torch) * sine)
            key_rotary = (key_rotary * cosine) + (_rotate_half(key_rotary, torch) * sine)
            if dimension == query.shape[-1]:
                return query_rotary, key_rotary
            return (
                torch.cat((query_rotary, query[..., dimension:]), dim=-1),
                torch.cat((key_rotary, key[..., dimension:]), dim=-1),
            )

    return Primitive()


def _parameter_name(role: str) -> str:
    return role.replace("-", "_")


def _enum_case(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value))
    raise ExportError("invalid_enum", repr(value))


def _layer_indices(value: Any) -> frozenset[int]:
    if not isinstance(value, dict) or set(value) != {"layerIndices"}:
        raise ExportError("unsupported_condition", repr(value))
    payload = value["layerIndices"]
    if isinstance(payload, dict):
        payload = payload.get("_0")
    if not isinstance(payload, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) and index >= 0
        for index in payload
    ):
        raise ExportError("unsupported_condition", repr(value))
    return frozenset(payload)


def _activate(value: Any, activation: str, torch: Any) -> Any:
    if activation in {"silu", "swish"}:
        return torch.nn.functional.silu(value)
    if activation == "gelu":
        return torch.nn.functional.gelu(value)
    if activation == "relu":
        return torch.nn.functional.relu(value)
    if activation == "sigmoid":
        return torch.sigmoid(value)
    raise RuntimeError(f"Unsupported activation {activation}")


def _validate_lowering_contract(region: dict[str, Any], path: str) -> None:
    supported_primitives = {
        "attention",
        "layer_norm",
        "layer_scale",
        "linear",
        "mlp",
        "moe",
        "output_head",
        "rms_norm",
        "short_conv",
        "state_space",
        "token_embedding",
    }
    for index, operation in enumerate(region["operations"]):
        operation_path = f"{path}.operations[{index}]"
        kind = operation["kind"]
        tag = kind["tag"]
        if tag == "primitive":
            primitive = kind["primitive"]
            opcode = primitive["opcode"]
            if opcode not in supported_primitives:
                raise ExportError(
                    "unsupported_lowering",
                    f"{operation_path}: primitive {opcode!r}",
                )
            _validate_primitive_contract(
                opcode,
                primitive["attributes"],
                operation["parameterBindings"],
                operation["stateBindings"],
                operation_path,
            )
            continue
        if tag == "residual":
            strategy = _enum_case(kind["strategy"])
            if strategy != "add":
                raise ExportError(
                    "unsupported_lowering",
                    f"{operation_path}: residual strategy {strategy!r}",
                )
            _validate_lowering_contract(kind["body"], f"{operation_path}.body")
            continue
        if tag == "parallel":
            merge = _enum_case(kind["merge"])
            if merge != "add":
                raise ExportError(
                    "unsupported_lowering",
                    f"{operation_path}: parallel merge {merge!r} has no axis contract",
                )
            for branch_index, branch in enumerate(kind["branches"]):
                _validate_lowering_contract(
                    branch,
                    f"{operation_path}.branches[{branch_index}]",
                )
            continue
        if tag == "repeating":
            _validate_lowering_contract(kind["body"], f"{operation_path}.body")
            continue
        if tag == "conditional":
            _layer_indices(kind["condition"])
            _validate_lowering_contract(kind["then"], f"{operation_path}.then")
            _validate_lowering_contract(kind["else"], f"{operation_path}.else")
            continue
        raise ExportError("unsupported_lowering", f"{operation_path}: {tag!r}")


def _validate_primitive_contract(
    opcode: str,
    attributes: dict[str, Any],
    bindings: list[dict[str, Any]],
    state_bindings: list[dict[str, Any]],
    path: str,
) -> None:
    roles = {binding["role"] for binding in bindings}
    state_roles = {binding["role"] for binding in state_bindings}
    required_roles, allowed_roles = _parameter_roles(opcode, attributes)
    missing_roles = required_roles - roles
    unknown_roles = roles - allowed_roles
    if missing_roles:
        raise ExportError(
            "unsupported_lowering",
            f"{path}: missing parameter roles {sorted(missing_roles)!r}",
        )
    if unknown_roles:
        raise ExportError(
            "unsupported_lowering",
            f"{path}: unused parameter roles {sorted(unknown_roles)!r}",
        )

    allowed_state_roles: set[str] = set()
    if opcode == "attention":
        allowed_state_roles = {"key_cache", "value_cache"}
        if state_roles and state_roles != allowed_state_roles:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: attention state requires key_cache and value_cache",
            )
    elif opcode == "short_conv":
        allowed_state_roles = {"conv_cache"}
    elif opcode == "state_space":
        allowed_state_roles = {"conv_cache", "recurrent_state"}
        if state_roles and state_roles != allowed_state_roles:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: state-space state requires conv_cache and recurrent_state",
            )
    unknown_state_roles = state_roles - allowed_state_roles
    if unknown_state_roles:
        raise ExportError(
            "unsupported_lowering",
            f"{path}: unused state roles {sorted(unknown_state_roles)!r}",
        )

    if opcode == "mlp":
        activation = _enum_case(attributes["activation"])
        gating = _enum_case(attributes["gating"])
        if activation not in {"gelu", "relu", "silu", "swish"}:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: MLP activation {activation!r}",
            )
        if gating not in {"none", "glu", "geglu", "swiglu"}:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: MLP gating {gating!r}",
            )
    if opcode == "moe":
        expert_count = attributes["expertCount"]
        experts_per_token = attributes["expertsPerToken"]
        if expert_count <= 0 or not 0 < experts_per_token <= expert_count:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: invalid MoE expert counts",
            )
        gate_kind = _enum_case(attributes["gateKind"])
        if gate_kind not in {"topK", "sigmoidTopK"}:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: MoE gate kind {gate_kind!r}",
            )
        expert_mlp = attributes["expertMLP"]
        unsupported = []
        if expert_mlp["inputSize"] != expert_mlp["outputSize"]:
            unsupported.append("different input/output sizes")
        if _enum_case(expert_mlp["activation"]) not in {"silu", "swish"}:
            activation = _enum_case(expert_mlp["activation"])
            unsupported.append(f"activation {activation!r}")
        if _enum_case(expert_mlp["gating"]) != "swiglu":
            unsupported.append(f"gating {_enum_case(expert_mlp['gating'])!r}")
        if expert_mlp["bias"]:
            unsupported.append("projection bias")
        if unsupported:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: MoE expert MLP {', '.join(unsupported)}",
            )
    if opcode == "state_space":
        variant = attributes["variant"]
        if variant != "gated_deltanet":
            raise ExportError(
                "unsupported_lowering",
                f"{path}: state-space variant {variant!r}",
            )
        if attributes["computeDType"] != "float32":
            raise ExportError(
                "unsupported_lowering",
                f"{path}: state-space compute dtype {attributes['computeDType']!r}",
            )
        if attributes["numHeads"] % attributes["groupCount"] != 0:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: state-space head count must be divisible by group count",
            )
        if attributes["normEpsilon"] <= 0:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: state-space norm epsilon must be positive",
            )
    if opcode == "attention":
        unsupported = {
            "sharedKeyValueSourceLayerIndex": attributes.get(
                "sharedKeyValueSourceLayerIndex"
            ),
            "valueNorm": attributes.get("valueNorm"),
            "window": attributes.get("window"),
        }
        enabled = [name for name, value in unsupported.items() if value is not None]
        rope = attributes.get("rope")
        if rope is not None and rope.get("scaling") is not None:
            enabled.append("scaled RoPE")
        output_gate = attributes.get("outputGate")
        if output_gate is not None and _enum_case(output_gate) != "sigmoidPackedInQProj":
            enabled.append(f"output gate {_enum_case(output_gate)}")
        qk_norm = attributes.get("qkNorm")
        if qk_norm is not None and _enum_case(qk_norm) not in {
            "none",
            "rmsNorm",
            "rmsNormUnitOffset",
        }:
            enabled.append(f"QK norm {_enum_case(qk_norm)}")
        value_source = attributes.get("valueProjectionSource")
        if value_source is not None and _enum_case(value_source) != "dedicatedProjection":
            enabled.append(f"value source {_enum_case(value_source)}")
        if enabled:
            raise ExportError(
                "unsupported_lowering",
                f"{path}: attention features {', '.join(enabled)}",
            )


def _parameter_roles(
    opcode: str,
    attributes: dict[str, Any],
) -> tuple[set[str], set[str]]:
    if opcode == "token_embedding":
        return {"embedding_table"}, {"embedding_table"}
    if opcode == "rms_norm":
        roles = {"scale"} if attributes.get("withScale", True) else set()
        return roles, roles
    if opcode == "layer_norm":
        if not attributes["affine"]:
            return set(), set()
        return {"scale"}, {"scale", "bias"}
    if opcode == "layer_scale":
        return {"layer_scalar"}, {"layer_scalar"}
    if opcode == "linear":
        roles = {"weight"}
        if attributes["bias"]:
            roles.add("bias")
        return roles, roles
    if opcode == "mlp":
        roles = {"up_proj", "down_proj"}
        if _enum_case(attributes["gating"]) != "none":
            roles.add("gate_proj")
        if attributes["bias"]:
            roles.update({"up_proj_bias", "down_proj_bias"})
            if "gate_proj" in roles:
                roles.add("gate_proj_bias")
        return roles, roles
    if opcode == "moe":
        roles = {"router"}
        for expert_index in range(attributes["expertCount"]):
            roles.update(
                {
                    f"expert_{expert_index}_gate_proj",
                    f"expert_{expert_index}_up_proj",
                    f"expert_{expert_index}_down_proj",
                }
            )
        if attributes["useExpertBias"]:
            roles.add("expert_bias")
        return roles, roles
    if opcode == "short_conv":
        required = {"in_proj", "conv_weight", "out_proj"}
        optional = {"in_proj_bias", "conv_bias", "out_proj_bias"}
        return required, required | optional
    if opcode == "state_space":
        roles = {
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
            "scale",
            "conv_weight",
            "dt_bias",
            "A_log",
        }
        return roles, roles
    if opcode == "attention":
        roles = {"q_proj", "k_proj", "o_proj"}
        value_source = _enum_case(
            attributes.get("valueProjectionSource", "dedicatedProjection")
        )
        if value_source == "dedicatedProjection":
            roles.add("v_proj")
        qk_norm = attributes.get("qkNorm")
        if qk_norm is not None and _enum_case(qk_norm) != "none":
            roles.update({"q_layernorm", "k_layernorm"})
        if attributes.get("bias", False):
            roles.update({"q_proj_bias", "k_proj_bias", "o_proj_bias"})
            if "v_proj" in roles:
                roles.add("v_proj_bias")
        return roles, roles
    if opcode == "output_head":
        roles = {"weight"}
        if attributes["bias"]:
            roles.add("bias")
        return roles, roles
    raise ExportError("unsupported_lowering", f"primitive {opcode!r}")


def _rotate_half(value: Any, torch: Any) -> Any:
    midpoint = value.shape[-1] // 2
    if midpoint == 0:
        raise ExportError("invalid_rope_dimension", str(value.shape[-1]))
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def _repeat_kv(value: Any, repetitions: int) -> Any:
    if repetitions == 1:
        return value
    batch, heads, sequence, dimension = value.shape
    return value[:, :, None, :, :].expand(
        batch, heads, repetitions, sequence, dimension
    ).reshape(batch, heads * repetitions, sequence, dimension)
