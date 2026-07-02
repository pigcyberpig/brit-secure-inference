import argparse

import torch

import crypten
import crypten.communicator as comm
from crypten.nn.module import GELU


X_SOFTMAX = torch.tensor(
    [
        [-3.0, -1.0, 0.0, 1.0, 3.0],
        [2.5, 0.5, -0.5, -1.5, 4.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
X_GELU = torch.linspace(-4.0, 4.0, steps=17).reshape(1, -1)


SHAFT_CONFIG = {
    "functions.softmax_method": "ode",
    "functions.gelu_method": "fourier",
}


def _metrics(output, reference):
    diff = (output - reference).abs()
    return diff.max().item(), diff.mean().item()


def _print_result(name, output, reference):
    max_abs, mean_abs = _metrics(output, reference)
    print(f"{name}: max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}")
    print(f"{name}.output={output}")


def run_case(label, override):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()

    with crypten.cfg.temp_override(override):
        enc_softmax = crypten.cryptensor(X_SOFTMAX, src=0)
        softmax_out = enc_softmax.softmax(dim=-1).get_plain_text()

        enc_gelu = crypten.cryptensor(X_GELU, src=0)
        gelu_tensor_out = enc_gelu.gelu().get_plain_text()

        gelu_module = GELU(approximate=b"none").encrypt()
        gelu_module_out = gelu_module(crypten.cryptensor(X_GELU, src=0)).get_plain_text()

        if rank == 0:
            print(
                f"\nCASE {label}: world_size={world_size}, "
                f"softmax={crypten.cfg.functions.softmax_method}, "
                f"gelu={crypten.cfg.functions.gelu_method}, "
                f"erf={crypten.cfg.functions.erf_method}"
            )
            _print_result(
                f"{label}.softmax",
                softmax_out,
                torch.softmax(X_SOFTMAX, dim=-1),
            )
            print(f"{label}.softmax_row_sums={softmax_out.sum(dim=-1)}")
            _print_result(
                f"{label}.gelu_tensor",
                gelu_tensor_out,
                torch.nn.functional.gelu(X_GELU),
            )
            _print_result(
                f"{label}.gelu_module",
                gelu_module_out,
                torch.nn.functional.gelu(X_GELU),
            )


def run_all_cases():
    if comm.get().get_rank() == 0:
        print(f"crypten={crypten.__file__}")
        print(
            "default_config="
            f"softmax:{crypten.cfg.functions.softmax_method}, "
            f"gelu:{crypten.cfg.functions.gelu_method}, "
            f"erf:{crypten.cfg.functions.erf_method}"
        )

    run_case("default", {})
    run_case("shaft_approx", SHAFT_CONFIG)


@crypten.mpc.run_multiprocess(world_size=2)
def run_multiprocess_cases():
    run_all_cases()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="run as a 2-party CrypTen MPC test",
    )
    args = parser.parse_args()

    torch.set_printoptions(precision=6, sci_mode=False)

    if args.multiprocess:
        run_multiprocess_cases()
    else:
        crypten.init()
        run_all_cases()
        crypten.uninit()


if __name__ == "__main__":
    main()
