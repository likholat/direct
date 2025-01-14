.. highlight:: shell

=========
Inference
=========

After training a model, you can use the ``direct predict`` command to perform inference.

To perform inference on a single machine run the following code block in your linux machine:

.. code-block:: bash

    $ direct predict <data_root> <output_directory> --cfg <cfg_path_or_url> --checkpoint <checkpoint_path_or_url> --num-gpus <num_gpus> [ --cfg <cfg_filename>.yaml --other-flags <other_flags>]
                  
To predict using multiple machines run the following code (one command on each machine):

.. code-block:: bash

    (machine0)$ direct predict <data_root> <output_directory> --cfg <cfg_path_or_url> --checkpoint <checkpoint_path_or_url> --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ direct predict <data_root> <output_directory> --cfg <cfg_path_or_url> --checkpoint <checkpoint_path_or_url> --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]

The ``cfg_path_or_url`` should point to a configuration file that includes all the model parameters used for the trained model checkpoint ``checkpoint_path_or_url`` and should also include an inference configuration as follows:

.. code-block:: yaml

    inference:
  dataset:
    name: InferenceDataset
    lists: ...
    transforms:
      masking:
        ...
      ...
    text_description: <inference_description>
    ...
  batch_size: <batch_size>
  ...
