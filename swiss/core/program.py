"""The Metal device and step programs."""

from functools import cache

from carbon import Device


def allocate(device, sizes):
    """Allocate {name: bytes}, rounding each buffer up to a positive multiple of 16."""
    return {
        name: device.buffer(max(16, -(-size // 16) * 16))
        for name, size in sizes.items()
    }


@cache
def open_device():
    """The Carbon device shared by every model in the process."""
    return Device()


class Program:
    """Dispatches recorded once and submitted as one command buffer per run."""

    def __init__(self, device):
        self.device, self.steps, self.prepared = device, [], None

    def kernel(self, dispatch):
        if self.prepared is not None:
            raise RuntimeError("program is already prepared")
        self.steps.append(dispatch)

    def finish(self):
        self.prepared = self.device.prepare(self.steps)

    def submit(self, repetitions=1):
        """Submit `repetitions` copies of the dispatches without waiting."""
        if self.prepared is None:
            self.finish()
        return self.prepared.submit(repetitions)

    def run(self):
        """Submit and wait; return the completion."""
        completion = self.submit()
        completion.wait()
        return completion
