import numpy as np


# A function wrapper is a function that wraps another function and adds extra functionality before or after its execution.
# some black magic way to count the number of call of our function
# that dont involves global so that we can paralellize easier
def counted(f):  # takes a function and replace it by another
    def wrapper(*args, **kwargs):  # our new function is wrapper
        wrapper.calls += 1  # increase an atribute called calls
        return f(*args, **kwargs)  # and will return the result of the fonction

    wrapper.calls = 0  # at start calls =0
    wrapper.__name__ = f.__name__  # the name is the same
    return wrapper  # replace the function by the new one


def reset_counters():
    getObjFVal.calls = getObjFGrad.calls = getObjFHess.calls = 0


@counted
def getObjFVal(x, functionID):
    x, y = x[0], x[1]
    if functionID == 1:
        fval = 5 * (x + 1) ** 2 - 4 * x * y + 3 * y**2 - 2 * x + 7 * y + 2
    elif functionID == 2:
        fval = (
            0.2 * x**2
            + 0.3 * y**2
            - 0.2 * x * y
            - 15 * np.cos(0.7 * x)
            - 9 * np.sin(0.6 * y - 0.7 * x)
            + 3 * x
        )
    else:
        raise ValueError("Unknown functionID")

    return fval


@counted
def getObjFGrad(x, functionID):
    x, y = x[0], x[1]
    if functionID == 1:
        return np.array([10 * x - 4 * y + 8, -4 * x + 6 * y + 7])
    elif functionID == 2:
        return np.array(
            [
                0.4 * x
                - 0.2 * y
                + 10.5 * np.sin(0.7 * x)
                + 6.3 * np.cos(0.7 * x - 0.6 * y)
                + 3,
                -0.2 * x + 0.6 * y - 5.4 * np.cos(0.7 * x - 0.6 * y),
            ]
        )
    raise ValueError("Unknown functionID")


@counted
def getObjFHess(x, functionID):
    x, y = x[0], x[1]
    if functionID == 1:
        return np.array([[10.0, -4.0], [-4.0, 6.0]])
    elif functionID == 2:
        return np.array(
            [
                [
                    -4.41 * np.sin(0.7 * x - 0.6 * y) + 7.35 * np.cos(0.7 * x) + 0.4,
                    3.78 * np.sin(0.7 * x - 0.6 * y) - 0.2,
                ],
                [
                    3.78 * np.sin(0.7 * x - 0.6 * y) - 0.2,
                    0.6 - 3.24 * np.sin(0.7 * x - 0.6 * y),
                ],
            ]
        )
    raise ValueError("Unknown functionID")
