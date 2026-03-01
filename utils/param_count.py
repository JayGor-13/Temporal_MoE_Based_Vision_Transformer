def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active = total
    for m in model.modules():
        router = getattr(m, "router", None)
        if router is not None and getattr(router, "last_probs", None) is not None:
            # approximate active as top-k proportion in routed layers
            top_k = router.top_k
            n = router.num_experts
            active = int(active * (top_k / n + 0.5))
    return {"total_params": int(total), "active_params": int(active)}
