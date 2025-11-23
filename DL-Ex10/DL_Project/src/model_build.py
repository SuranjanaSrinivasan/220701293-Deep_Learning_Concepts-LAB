from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_dnn_model(
    input_dim: Union[int, Tuple[int, ...]],
    hidden_units: Sequence[int] = (64, 32),
    dropout_rate: float = 0.3,
    activation: str = 'relu',
    output_units: int = 1,
    output_activation: str = 'sigmoid',
    compile_model: bool = True,
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
    loss: Optional[str] = None,
    metrics: Optional[Sequence[str]] = ('accuracy',),
) -> tf.keras.Model:
    """Build and (optionally) compile a simple feed-forward Keras model.

    Args:
        input_dim: Number of input features, or a tuple for input shape.
        hidden_units: Sequence with sizes of hidden Dense layers (at least one).
        dropout_rate: Dropout rate placed after the first Dense layer.
        activation: Activation for hidden layers.
        output_units: Number of units in the output layer.
        output_activation: Activation for the output layer.
        compile_model: If True, the function will call `model.compile(...)` before returning.
        optimizer: Optimizer name or instance for compilation.
        loss: Loss name; if None a sensible default is chosen.
        metrics: Sequence of metric names for compilation.

    Returns:
        A Keras `Model` instance.
    """

    # validate hidden_units
    if not hidden_units:
        raise ValueError('hidden_units must contain at least one integer')

    # normalize input shape
    if isinstance(input_dim, int):
        input_shape = (input_dim,)
    else:
        input_shape = tuple(input_dim)

    model = Sequential()

    # first layer must specify input_shape
    model.add(Dense(hidden_units[0], activation=activation, input_shape=input_shape))
    if dropout_rate and dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation))

    model.add(Dense(output_units, activation=output_activation))

    if compile_model:
        # choose a reasonable default loss when not provided
        if loss is None:
            if output_units == 1 and output_activation in ('sigmoid',):
                loss = 'binary_crossentropy'
            else:
                loss = 'mse'

        model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics) if metrics else None)

    return model


__all__ = ["build_dnn_model"]
