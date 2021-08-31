## Recurrent Intensity Model Experiments

Repository to reproduce the experiments in the paper:

Yifei Ma, Ge Liu Anoop Deoras. Recurrent Intensity Modeling for User Recommendation and Online Matching. ICML 2021 Time-Series Workshop.
[paper link1](http://roseyu.com/time-series-workshop/submissions/2021/TSW-ICML2021_paper_47.pdf)
[paper link2](https://www.amazon.science/publications/recurrent-intensity-modeling-for-user-recommendation-and-online-matching)


## Getting Started

1. Clone with submodules `git clone --recursive <this repository>`
2. Install via `pip install -e .`
3. Add data to [data](data) folder. See detailed instructions therein.
4. Run experiment as
    ```
    from rim_experiments import main, plot_results
    self = main("prepare_ml_1m_data")
    fig = plot_results(self)
    ```
4. To extend to other datasets, see example in [prepare_synthetic_data](src/rim_experiments/dataset/__init__.py)
5. The provided examples are tested in [test](test).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

