import torch
def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m-1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out

def flag_sbap(model_forward, perturb_shape, step_size, m, optimizer, device) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    (regression_loss_IC50, regression_loss_K), \
    (affinity_pred_IC50, affinity_pred_K), \
    (affinity_IC50, affinity_K) = forward(perturb)

    loss = regression_loss_IC50 + regression_loss_K
    loss /= m

    for _ in range(m-1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        (regression_loss_IC50, regression_loss_K), \
        (affinity_pred_IC50, affinity_pred_K), \
        (affinity_IC50, affinity_K) = forward(perturb)

        loss = regression_loss_IC50 + regression_loss_K

        loss /= m

    loss.backward()
    optimizer.step()

    return loss
