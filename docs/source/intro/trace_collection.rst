Trace Collection
================

Trace collection in PyTorch is enabled by wrapping the training/inference loop
in a ``profile`` context. A couple of useful options to know about are
``tracing schedule`` and ``trace handler``. The `tracing schedule` allows the
user to specify how many steps we can skip, wait, warmup the profiler, record
the activity and finally how many times to repeat the process. During the
warmup, the profiler is running but no events are being recorded hence there is
no profiling overhead. The `trace handler` allows to specify the output folder
along with the option to gzip the trace file. Given that trace files can easily
run into hundreds of MBs this is useful to have.

The ``profile`` context also gives options to record either or both CPU and GPU
events using the activities argument. Users can also record the shapes of the
tensors with ``record_shapes`` argument and collect the python call stack with
the ``with_stack`` argument. The ``with_stack`` argument is especially helpful in
connecting the trace event to the source code, which enables faster debugging.
The ``profile_memory`` option allows tracking tensor memory allocations and
deallocations.

To profile, wrap the code in the ``profile`` context manager as shown below.

.. code-block:: python
    :linenos:
    :emphasize-lines: 19

    from torch.profiler import profile, schedule, tensorboard_trace_handler

    tracing_schedule = schedule(skip_first=5, wait=5, warmup=2, active=2, repeat=1)
    trace_handler = tensorboard_trace_handler(dir_name="traces", use_gzip=True)

    NUM_EPOCHS = 5 # arbitrary number of epochs to profile

    with profile(
      activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
      schedule = tracing_schedule,
      on_trace_ready = trace_handler,
      profile_memory = True,
      record_shapes = True,
      with_stack = True
    ) as prof:

        for _ in range(NUM_EPOCHS):
          for step, batch_data in enumerate(data_loader):
              train(batch_data)
              prof.step()

Line 17 in the code snippet above signals to the profiler that a training
iteration has completed.
