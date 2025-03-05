# LLM4TS

Welcome to the official codebase of LLM4TS.
This project is based on research that has been accepted for publication in the ACM Transactions on Intelligent Systems and Technology (TIST) 2025.

# Usage

1. Install Python 3.8, and use `requirements.txt` to install the dependencies
   ```
   pip install -r requirements.txt
   ```
2. Place all datasets in the `dataset` folder. The datasets can be downloaded from [this link](https://drive.google.com/drive/folders/1vE0ONyqPlym2JaaAoEe0XNDR8FS_d322).
3. Place the [GPT-2 model from Hugging Face](https://huggingface.co/gpt2/tree/main) into the `LLM/gpt2` directory.
4. To execute the script with configuration settings passed via argparse, use:
   ```
   python main.py --...
   ```
   Alternatively, if you prefer to use locally defined parameters to overwrite args for faster experimentation iterations, run:
   ```
   python main.py --overwrite_args
   ```
5. Please refer to `exp_settings_and_results` to see all the experiments' settings and corresponding results.

# Citation

If you find value in this repository, we kindly ask that you cite our paper.

```
@article{chang2023llm4ts,
  title={LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs},
  author={Chang, Ching and Peng, Wen-Chih and Chen, Tien-Fu},
  journal={arXiv preprint arXiv:2308.08469},
  year={2023}
}
```

# Contact

If you have any questions or suggestions, please reach out to Ching Chang at [blacksnail789521@gmail.com](mailto:blacksnail789521@gmail.com), or raise them in the 'Issues' section.

# Acknowledgement

This library was built upon the following repositories:

* Time Series Library (TSlib): [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
