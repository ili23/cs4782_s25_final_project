# CS4782 Spring 2025 Final Project

## Reimplementation of A Time Series is Worth 64 Words: Long-term Forecasting with Transformers

Group 33: Iram Liu (il233), Jeffrey Xiang (jjx5), Wen Chen (wc467), Ajay Sachdev (aks262), Nick Hsieh (njh79)

## Introduction

We reimplemented the findings that were found in the paper "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" where it aimed to improve the performance of transfomer-based models over linear models when it came to predicting time series datasets. The two main things that the paper introduced were patching and channel independence. Patching is where time series data is segmented into subseries-level patches that serve as input tokens to the Transformer allowing the model to capture local semantic information. Channel independence is where a multivariate time series is broken into multiple univariate time series and fed individually into the transformer.

## Chosen Result

We were aiming to reproduce the results of applying supervised PatchTST/64 over five datasets as demonstrated in the paper. If you take a look at the chart below, we want to replicate the rows labeled with 96 the prediction length or 24 for the ILI dataset.
![Figure 1](/PaperResults.png)

## Github Contents

In our /data folder we have the dataloader setup, within /models we have the model architecture implementation, inside the /poster folder we have a copy of our poster and the /report has a pdf of our project report. Inside the /results you will find the results that we achieved. Additionally, within every folder in results there is a log folder inside them and where you can see the he train/val/test losses we obtained please run run “tensorboard --logdir .” within the respective log directory.

## Re-implementation Details

Our re-implementation of supervised PatchTST/64 is nearly identical to the model architecture described in the paper. One key feature we used was channel splitting which turned a multivariate time series into multiple univariate time series. Another was instance norm and patching (patch length P=16) which is applied on each univariate time series. Then, resulting patches are mapped to the Transformer latent space, and a learnable positional encoding is added. The tokens are fed into the Transformer encoder consisting of 3 layers. Finally, the outputs of the transformer are fed through a flatten layer and each channel is concatenated to recreate a multivariate prediction. The predictions are fed through the MSE loss function to train the model. One key challenge that we ran into was that on smaller datasets limiting us to using only 4 heads in the encoder layer instead of the 16 heads used in other PatchTST models.

## Reproduction Steps

First download the datasets from the link below and drop the files into the /data folder

Dataset: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Then install the dependencies from requirements.txt

Then finally to run the model and train it run the file supervised.py by being in the cs4782_s25_final_project directory with the following command

python supervised.py

## Results/Insights

![Figure 2](/OurResults.png)
As you can see from the results of our re-implementation results shows similar findings. We were able to show that our re-implementation of PatchTST/64 produced results close to the ones described in the paper and did better than past transformers across almost all the datasets. Our re-implementation comes very close to beating the results produced by DLinear. We attribute the slightly lower performance of our PatchTST/64 to compute limitations that we faced during training. On the Traffic and Electricity models, we trained for only 3 epochs compared to the 100 epochs mentioned in the paper. For the ETTh1 and ETTm1 models, we trained for only 25 epochs compared to the 100 epochs in the paper. As for the ILI model, we trained the same number of epochs as the paper (100 epochs). With more training you can expect the results to get closer to the ones shown in the paper and even beat the results produced by DLinear.

## Conclusion

Through our re-implementation we learned a couple of things, firstly RAM storage is a large concern when training models on small GPUs. We noticed that training the Transformer with even a few encoder layers demanded significant computation due to the sheer number of parameters to learn. Therefore, this constrained our ability to train Transformer models. Secondly, when we began training the models, we noticed that regularization techniques, such as dropout, early stopping, and channel-independence, noticeably increased the performance of our models. We were surprised by the extent to which these techniques helped reduce testing loss.

## References

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. arXiv:2211.14730, 2023.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are Transformers Effective for Time Series Forecasting? arXiv preprint arXiv:2205.13504, 2022.

Trindade, A. (2015). ElectricityLoadDiagrams20112014 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 11106–11115)

## Acknowledgements

This final project was completed by Iram Liu, Jeffrey Xiang, Wen Chen, Ajay Sachdev, and Nick Hsieh as part of the coursework for CS 4782: Introduction to Deep Learning.
