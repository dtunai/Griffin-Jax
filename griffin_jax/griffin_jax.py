import jax
import numpy as np
import jax.numpy as jnp

from jax import random, jit
from jax import numpy as np
from flax import linen as nn
from scipy.special import expit


class FeedForward(nn.Module):
    """
    FNN module for performing MLP operations

    Args:
        dim (int): Dimensionality of the input and output features
        mult (int): Multiplier for the intermediate feature dimension
        post_act_ln (bool): Flag indicating whether to apply Layer Normalization after activation
        dropout (float): Dropout rate for regularization

    Attributes:
        dense1 (nn.Linear): First fully connected layer
        dropout_layer (nn.Dropout): Dropout layer for regularization
        dense2 (nn.Linear): Second fully connected layer

    Methods:
        setup(): Initialize and set up the layers with appropriate initialization
        __call__(x): Forward pass through the network
    """

    dim: int
    mult: int
    post_act_ln: bool
    dropout: float

    def setup(self):
        self.dense1 = nn.Linear(
            features=self.dim * self.mult,
            kernel_init=lambda key: nn.initializers.xavier_uniform()(
                key, (self.dim * self.mult, self.dim)
            ),
        )
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        self.dense2 = nn.Dense(
            features=self.dim,
            kernel_init=lambda key: nn.initializers.xavier_uniform()(
                key, (self.dim * self.mult, self.dim)
            ),
        )

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dropout_layer(x)
        x = self.dense2(x)
        return x


class RGLRU(nn.Module):
    dim: int
    mult: int

    def setup(self):
        self.c = 8
        key = random.PRNGKey(0)
        self.Wa = self.param(
            "Wa", nn.initializers.xavier_uniform(), (self.dim * self.mult, self.dim)
        )
        self.Wx = self.param(
            "Wx", nn.initializers.xavier_uniform(), (self.dim * self.mult, self.dim)
        )
        self.ba = self.param("ba", nn.initializers.zeros, (self.dim * self.mult,))
        self.bx = self.param("bx", nn.initializers.zeros, (self.dim * self.mult,))
        self.Lambda = self.param(
            "Lambda",
            nn.initializers.uniform(minval=expit(0.9), maxval=expit(0.999)),
            (self.dim * self.mult,),
        )

    @nn.compact
    def __call__(self, x):
        batch_size, sequence_length, _ = x.shape
        ht = jnp.zeros((batch_size, self.dim * self.mult), device=x.device)
        y = []

        for t in range(sequence_length):
            xt = x[:, t, :]
            rt = jax.nn.sigmoid(jnp.dot(xt, self.Wa) + self.ba)
            it = jax.nn.sigmoid(jnp.dot(xt, self.Wx) + self.bx)
            a = jax.nn.sigmoid(self.Lambda)
            at = a / self.c**rt
            ht = at * ht + ((1 - at**2) ** 0.5) * (it * xt)
            y.append(ht[jnp.newaxis, :])

        y = jnp.concatenate(y, axis=1)
        return y


class RMSNorm(nn.Module):
    dim: int

    def setup(self):
        self.scale = self.dim**0.5
        self.g = self.param("g", nn.initializers.ones, (self.dim,))

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (jax.numpy.ndarray): The input tensor.

        Returns:
            jax.numpy.ndarray: The normalized output tensor.

        """
        scale = self.scale
        g = self.g
        print("G: ", g.shape)
        normalized_x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)[:, :, :1]
        print("After slicing: ", normalized_x.shape)
        return normalized_x * (self.scale * self.g[: normalized_x.shape[-1]])


def output_head(x, num_tokens, dim):
    """
    Applies a linear transformation followed by softmax activation to the input tensor

    Args:
        x (jax.interpreters.xla.DeviceArray): Input tensor of shape (batch_size, dim)
        num_tokens (int): Number of output tokens
        dim (int): Dimension of the input tensor

    Returns:
        jax.interpreters.xla.DeviceArray: Output tensor after applying linear transformation and softmax activation
    """
    x = RMSNorm(dim)(x)
    x = nn.Dense(dim, num_tokens)(x)
    x = jax.nn.softmax(x, axis=-1)

    return x


class GriffinResidualBlock(nn.Module):
    dim: int
    depth: int
    mlp_mult: int
    dropout: float = 0.1
    heads: int = 8

    def setup(self):
        self.norm = RMSNorm(self.dim)
        self.mlp = FeedForward(self.dim, self.mlp_mult, False, self.dropout)
        self.lru = RGLRU(
            self.dim,
            mult=4,
        )

    @nn.compact
    def __call__(self, x):
        mlp_mult = 4

        b, s, d = x.shape
        skip = x

        x = self.norm(x)
        print("Normalized X Shape: ", x.shape)

        print(self.dim, mlp_mult, self.dropout, self.heads)

        linear_1, linear_2 = nn.Dense(features=self.dim)(x), nn.Dense(features=self.dim)(x)
        print("Shape of Linear 1:", linear_1.shape, "Shape of Linear 2:", linear_2.shape)

        linear_1 = nn.Conv(features=self.dim, kernel_size=(3,), padding="SAME")(linear_1)
        print("Shape of Linear 1 Conv: ", linear_1.shape)

        linear_2 = nn.gelu(linear_2)
        print("Linear 2 Shape GeLU: ", linear_2.shape)

        # linear_1 = np.repeat(linear_1, 10, axis=-1)
        # print("Linear 1 Reshaped: ", linear_1.shape)

        # Element wise multiplication to merge the paths
        x = linear_1 * linear_2

        print("Shape of X after Temporal Mixing Block:", x.shape)

        # Skip
        x += skip

        print("Shape of X after Skip:", x.shape)

        # Skip2
        skip2 = x

        # Norm
        x = self.norm(x)

        # Feed Forward
        x = self.mlp(x)

        print("Shape of X Final:", x.shape)

        return x + skip2


class Griffin(nn.Module):
    """
    Griffin module for performing Griffin Residual Network operations

    Args:
        dim (int): Dimension of the input tensor
        depth (int): Number of residual blocks in the network
        mlp_mult (int): Multiplier for the hidden dimension of the MLP layers
        dropout (float, optional): Dropout probability. Defaults to 0.1
        heads (int, optional): Number of attention heads. Defaults to 8
        filter (int, optional): Filter size for the convolutional layers.

    Attributes:
        dim (int): Dimension of the input tensor
        depth (int): Number of residual blocks in the network
        mlp_mult (int): Multiplier for the hidden dimension of the MLP layers
        dropout (float): Dropout probability
        heads (int): Number of attention heads
        filter (int): Filter size for the convolutional layers
        layers (nn.ModuleList): List of GriffinResidualBlock layers

    """

    dim: int
    num_tokens: int
    seq_len: int
    depth: int
    mlp_mult: int
    dropout: float
    heads: int = 8

    def setup(self):
        self.emb = nn.Embed(
            self.dim,
            self.num_tokens,
        )

        self.norm = RMSNorm(self.dim)
        self.layers = [
            GriffinResidualBlock(self.dim, self.mlp_mult, self.dropout) for _ in range(self.depth)
        ]

    @nn.compact
    def __call__(self, x):
        """
        Applies the Griffin transformation to the input tensor

        Args:
            x (jax.interpreters.xla.DeviceArray): The input tensor

        Returns:
            jax.interpreters.xla.DeviceArray: The output tensor after applying the Griffin transformation

        """
        x = self.emb(x)
        x = self.norm(x)

        for layer in self.layers:
            x = layer(x) + x

        return output_head(x, self.num_tokens, self.dim)
