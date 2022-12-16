import torch
import torch.nn as nn
import torch.nn.functional as F


def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(
        device
    )
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0.0, 1.0)
    # Return perturbed samples
    return x_adv

# def generation_loss(mod, inp, targ):
#     op = mod(inp)
#     loss = ch.nn.CrossEntropyLoss(reduction='none')(op, targ)
#     return loss, None


# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model, device, data, lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out, lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if rand_start == True:
        x_adv = random_noise_attack(model, device, dat, eps)
    else:
        x_adv = x_nat

    # Make sure the sample is projected into original distribution bounds [0,1]
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # Iterate over iters
    for epoch in range(iters):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        gradient = gradient_wrt_data(model, device, x_adv, lbl)
        # Perturb the image using the gradient
        x_perturb = x_adv - alpha * gradient.sign()
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        perturb = torch.clamp((x_perturb - x_nat), min=-eps, max=+eps)
        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        x_adv = torch.clamp((x_nat + perturb), 0.0, 1.0)

    # Return the final perturbed samples
    return x_adv


def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float

    # HINT: FGSM is a special case of PGD
    x_adv = PGD_attack(
        model, device, dat, lbl, eps, alpha=eps, iters=1, rand_start=False
    )
    return x_adv


def rFGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    
    # clip image_noise to make sure it falls in [0,1]
    #image_noise = torch.clamp(image_noise, 0, 1)

    # HINT: rFGSM is a special case of PGD
    x_adv = PGD_attack(
        model,
        device,
        dat,
        lbl,
        eps,
        alpha = eps,
        iters=1,
        rand_start=True,
    )
    return x_adv


def FGM_L2_attack(model, device, dat, lbl, eps, iters):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    for i in range(iters):
      # Compute gradient w.r.t. data
      gradient = gradient_wrt_data(model, device, x_nat, lbl)
      # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
      # HINT: Flatten gradient tensor first, then compute L2 norm
      gradient_flattened = gradient.view(x_nat.size(0), -1)
      l2_of_grad = torch.linalg.norm(gradient_flattened, dim=1, ord=2)
      # Perturb the data using the gradient
      # HINT: Before normalizing the gradient by its L2 norm, use
      # torch.clamp(l2_of_grad, min=1e-12) to prevent division by 0
      l2_of_grad = torch.clamp(l2_of_grad, min=1e-12)
      grad_l2_normalized = gradient / l2_of_grad.view(x_nat.size(0), 1,1,1)
      # Add perturbation the data
      x_adv = x_nat - eps * grad_l2_normalized
      # Clip the perturbed datapoints to ensure we are in bounds [0,1]
      x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # Return the perturbed samples
    return x_adv
