import paddle


def get(opt, params, max_steps):
    if isinstance(params, list):
        params_group = params
    else:
        raise TypeError(f"`params` must be a list, got {type(params)}")

    if opt.lrp == "period_step":
        scheduler = paddle.optimizer.lr.StepDecay(opt.lr,
                                                  step_size=opt.lr_step,
                                                  gamma=opt.lr_rate)
    elif opt.lrp == "custom_step":
        scheduler = paddle.optimizer.lr.MultiStepDecay(opt.lr,
                                                       milestones=opt.lr_boundaries,
                                                       gamma=opt.lr_rate)
    elif opt.lrp == "plateau":
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(opt.lr,
                                                        factor=opt.lr_rate,
                                                        patience=opt.lr_patience,
                                                        threshold=opt.lr_min_delta,
                                                        cooldown=opt.cool_down,
                                                        min_lr=opt.lr_end)
    elif opt.lrp == "cosine":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(opt.lr,
                                                             T_max=max_steps,
                                                             eta_min=opt.lr_end)
    elif opt.lrp == "poly":
        scheduler = paddle.optimizer.lr.PolynomialDecay(opt.lr,
                                                        decay_steps=max_steps,
                                                        end_lr=opt.lr_end,
                                                        power=opt.power)
    else:
        raise ValueError

    if opt.optim == "sgd":
        optimizer_params = {"momentum": opt.sgd_momentum,
                            "weight_decay": opt.weight_decay,
                            "use_nesterov": opt.sgd_nesterov}
        optimizer = paddle.optimizer.Momentum(scheduler, parameters=params_group, **optimizer_params)
    elif opt.optim == "adam":
        optimizer_params = {"beta1": opt.adam_beta1,
                            "beta2": opt.adam_beta2,
                            "epsilon": opt.adam_epsilon,
                            "weight_decay": opt.weight_decay}
        optimizer = paddle.optimizer.Adam(scheduler, parameters=params_group, **optimizer_params)
    elif opt.optim == "adamw":
        optimizer_params = {"weight_decay": opt.weight_decay}
        optimizer = paddle.optimizer.AdamW(scheduler, parameters=params_group, **optimizer_params)
    else:
        raise ValueError(f"Not supported optimizer: {opt.optim}")

    return optimizer, scheduler
