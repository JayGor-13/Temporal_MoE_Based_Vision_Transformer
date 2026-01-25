from ..models.moe.hetero_moe_ffn import HeteroMoEFeedForward

def count_active_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_params = 0
    active_experts_by_layer = {}

    non_expert_params = 0
    total_expert_params_one_layer = 0

    # Separate non-expert, expert, and decoder params
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    encoder_base_params = sum(p.numel() for name, p in model.encoder.named_parameters() if 'moe' not in name and p.requires_grad)

    for name, module in model.encoder.named_modules():
        if isinstance(module, HeteroMoEFeedForward):
            router_params = sum(p.numel() for p in module.router.parameters() if p.requires_grad)
            if hasattr(module.router, "last_active_experts") and module.router.last_active_experts:
                active_set = set(module.router.last_active_experts)
                layer_active_params = router_params
                for e_id in active_set:
                    layer_active_params += sum(p.numel() for p in module.experts[e_id].parameters())
                active_params += layer_active_params
                active_experts_by_layer[name] = sorted(list(active_set))

    if active_params == 0: # Fallback if no forward pass has happened
        one_expert_size = sum(p.numel() for p in model.encoder.layers[0].moe.experts[0].parameters())
        router_params = sum(p.numel() for p in model.encoder.layers[0].moe.router.parameters())
        active_params_per_moe_layer = router_params + (model.encoder.layers[0].moe.router.top_k * one_expert_size)
        active_params = decoder_params + encoder_base_params + (len(model.encoder.layers) * active_params_per_moe_layer)
    else:
        active_params += decoder_params + encoder_base_params

    return int(active_params), int(total_params), active_experts_by_layer
    