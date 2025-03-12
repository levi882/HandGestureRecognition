# Mathematical Formulation of Hand Gesture Classifier Network

## Unified Network Representation

The hand gesture classifier can be described as a function $F$ that maps an input image $\mathbf{X} \in \mathbb{R}^{1 \times 128 \times 128}$ to a probability distribution over $N$ classes:

$$F(\mathbf{X}) = \text{softmax}(f_{\text{FC3}}(f_{\text{FC2}}(f_{\text{FC1}}(f_{\text{flatten}}(f_{\text{Conv5}}(f_{\text{Conv4}}(f_{\text{Conv3}}(f_{\text{Conv2}}(f_{\text{Conv1}}(\mathbf{X}))))))))))$$

Where each convolutional block function $f_{\text{Conv}i}$ is defined as:

$$f_{\text{Conv}i}(\mathbf{X}) = \text{Dropout}_i(\text{MaxPool}(\text{ReLU}(\text{BatchNorm}_i(\text{Conv2D}_i(\mathbf{X})))))$$

And each fully connected block function $f_{\text{FC}j}$ is defined as:

$$f_{\text{FC}j}(\mathbf{X}) = \text{Dropout}_j(\text{ReLU}(\text{BatchNorm}_j(\mathbf{W}_j\mathbf{X} + \mathbf{b}_j)))$$

The final layer (FC3) doesn't include BatchNorm, ReLU, or Dropout:

$$f_{\text{FC3}}(\mathbf{X}) = \mathbf{W}_3\mathbf{X} + \mathbf{b}_3$$

## Detailed Component Equations

### 1. Convolutional Operation

For each convolutional layer with input $\mathbf{X} \in \mathbb{R}^{C_{in} \times H_{in} \times W_{in}}$ and output $\mathbf{Y} \in \mathbb{R}^{C_{out} \times H_{out} \times W_{out}}$:

$$\mathbf{Y}_{c_{out}, h, w} = \sum_{c_{in}=1}^{C_{in}} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \mathbf{X}_{c_{in}, h+i, w+j} \cdot \mathbf{W}_{c_{out}, c_{in}, i, j} + \mathbf{b}_{c_{out}}$$

where:
- $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$ is the kernel tensor
- $\mathbf{b} \in \mathbb{R}^{C_{out}}$ is the bias vector
- $k_h, k_w$ are the kernel height and width (3×3 in our network)
- $C_{in}, C_{out}$ are the number of input and output channels

### 2. Batch Normalization

Given a mini-batch of inputs $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_m\}$, batch normalization performs:

$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{x}_i$$

$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{x}_i - \mu_\mathcal{B})^2$$

$$\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

$$\mathbf{y}_i = \gamma \hat{\mathbf{x}}_i + \beta$$

where $\gamma$ and $\beta$ are learnable parameters, and $\epsilon$ is a small constant for numerical stability.

During inference, the network uses the running statistics:

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mathrm{E}[\mathbf{x}]}{\sqrt{\mathrm{Var}[\mathbf{x}] + \epsilon}}$$

### 3. ReLU Activation Function

$$\text{ReLU}(\mathbf{x}) = \max(0, \mathbf{x})$$

Applied element-wise to each value in the input tensor.

### 4. Max Pooling

For a 2×2 max pooling with stride 2:

$$\text{MaxPool}(\mathbf{X})_{c, i, j} = \max_{0 \leq m, n < 2} \mathbf{X}_{c, 2i+m, 2j+n}$$

### 5. Dropout

During training, with dropout probability $p$:

$$\mathbf{y}_i = \begin{cases} 
\frac{\mathbf{x}_i}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}$$

During inference, dropout is not applied.

### 6. Flatten Operation

Transforms a tensor $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ into a vector $\mathbf{y} \in \mathbb{R}^{C \cdot H \cdot W}$:

$$\mathbf{y}_{c \cdot H \cdot W + h \cdot W + w} = \mathbf{X}_{c,h,w}$$

### 7. Fully Connected Layer

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{n_{out} \times n_{in}}$ and $\mathbf{b} \in \mathbb{R}^{n_{out}}$.

### 8. Softmax Function

For the output layer with logits $\mathbf{z} \in \mathbb{R}^N$:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{\mathbf{z}_i}}{\sum_{j=1}^{N} e^{\mathbf{z}_j}}$$

## Loss Function and Optimization

### Cross-Entropy Loss

For a batch of size $B$ with ground truth labels $\mathbf{y}$ and predicted probabilities $\hat{\mathbf{y}}$:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{B}\sum_{i=1}^{B} \sum_{c=1}^{N} \mathbf{y}_{i,c} \log(\hat{\mathbf{y}}_{i,c})$$

where $\mathbf{y}_{i,c}$ is a binary indicator (0 or 1) if class $c$ is the correct classification for sample $i$.

### Adam Optimizer

Adam optimizer maintains two moving averages for each parameter $\theta$:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla_\theta \mathcal{L}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla_\theta \mathcal{L}_t)^2$$

These are bias-corrected to account for the zero initialization:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

where:
- $\alpha$ is the learning rate (0.001 in our model)
- $\beta_1$ and $\beta_2$ are exponential decay rates (typically 0.9 and 0.999)
- $\epsilon$ is a small constant for numerical stability
- $\nabla_\theta \mathcal{L}_t$ is the gradient of the loss with respect to the parameters

With weight decay:

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \alpha \lambda \theta_{t-1}$$

where $\lambda$ is the weight decay coefficient (1e-5 in our model).

### Learning Rate Scheduler

The ReduceLROnPlateau scheduler monitors validation loss and reduces the learning rate when it plateaus:

$$\alpha_t = \begin{cases} 
\gamma \alpha_{t-1} & \text{if no improvement in validation loss for 'patience' epochs} \\
\alpha_{t-1} & \text{otherwise}
\end{cases}$$

where:
- $\gamma$ is the factor by which the learning rate is reduced (0.1 in our model)
- 'patience' is the number of epochs with no improvement after which learning rate will be reduced (5 in our model)

### Early Stopping

Training stops when validation loss fails to improve for a consecutive number of epochs:

$$\text{Stop training if } \text{counter} \geq \text{patience}$$

where counter is incremented when:

$$\text{val\_loss}_t > \text{best\_val\_loss} - \delta$$

and reset to 0 otherwise.

## Complete Network Dimensions

1. Input: $1 \times 128 \times 128$
2. Conv1 + BatchNorm + ReLU + MaxPool: $32 \times 64 \times 64$
3. Conv2 + BatchNorm + ReLU + MaxPool + Dropout: $64 \times 32 \times 32$
4. Conv3 + BatchNorm + ReLU + MaxPool: $128 \times 16 \times 16$
5. Conv4 + BatchNorm + ReLU + MaxPool + Dropout: $256 \times 8 \times 8$
6. Conv5 + BatchNorm + ReLU + MaxPool + Dropout: $512 \times 4 \times 4$
7. Flatten: $8192$
8. FC1 + BatchNorm + ReLU + Dropout: $1024$
9. FC2 + BatchNorm + ReLU + Dropout: $512$
10. FC3: $N$ (number of classes)
11. Softmax: $N$ (probability distribution)