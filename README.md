# Epochal Sawtooth Phenomenon
Qi Liu (liu_qi@tongji.edu.cn) and Wanjing Ma(mawanjing@tongji.edu.cn)

College of Transportation, Tongji University, Shanghai, P.R.China

Files for the results in the paper:

> Qi Liu and Wanjing Ma, 2025. The Epochal Sawtooth Phenomenon: Unveiling Training Loss Oscillations in Adam and Other Optimizers.
 - See `bert.py` for BERT model
 BERT small model benchmark settings:
 ```json
 {
    "TRAIN_DATA": "data/wikitext-103-xxl/wiki.train.tokens",
    "vocab_dir=": "vocab_500mb_30000.pkl",
    "NUM_CPUS": 8,
    "NUM_GPUS": 1,
    "batch_size": 128,
    "plot_every": 500,
    "total_num_epochs": 60,
    "num_steps_per_epoch": 7854,
    "total_num_steps": 471240,
    "num_batchs_used_for_val": 50,
    "total_num_batchs_in_val_dataset": 1587,
    "max_len": 90,
    "learning_rate": 0.0001,
    "weight_decay": 1e-05,
    "num_hiddens": 768,
    "ffn_num_hiddens": 1024,
    "num_heads": 6,
    "num_blks": 6,
    "dropout": 0.2,
    "clip_grad": false,
    "clip_grad_max_norm": 1.0,
    "use_warmup_lr_scheduler": false,
    "use_lr_scheduler": false,
    "lr_scheduler_step_size": 5000,
    "lr_scheduler_gamma": 1.0,
    "smoothing_coeff": 1.0
}
```
BERT tiny model benchmark settings:
```json
{
    "TRAIN_DATA": "data/wikitext-2-v1/wiki.train.tokens",
    "vocab_dir=": "vocab_500mb_30000.pkl",
    "NUM_CPUS": 8,
    "NUM_GPUS": 1,
    "batch_size": 32,
    "plot_every": 50,
    "total_num_epochs": 30,
    "num_steps_per_epoch": 1323,
    "total_num_steps": 39690,
    "num_batchs_used_for_val": 10,
    "total_num_batchs_in_val_dataset": 147,
    "data_loader_shuffle": true,
    "max_len": 90,
    "learning_rate": 0.0001,
    "weight_decay": 1e-05,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "num_hiddens": 256,
    "ffn_num_hiddens": 512,
    "num_heads": 2,
    "num_blks": 2,
    "dropout": 0.1,
    "clip_grad": false,
    "clip_grad_max_norm": 1.0,
    "use_warmup_lr_scheduler": false,
    "use_lr_scheduler": false,
    "lr_scheduler_step_size": 5000,
    "lr_scheduler_gamma": 0.8,
    "smoothing_coeff": 0.75
}
```
 - BERT model results
 Fig 2(a) MLM loss in BERT-small exhibits ESP.
 ![Fig 2(a) MLM loss in BERT-small exhibits ESP](img/reproduce.png)


 - See `bert-tiny-benchmark.ipynb` for BERT tiny model results

 Fig 2(b) Both MLM loss and NSP loss in BERT-tiny exhibit ESP.


 - See `replication-example-benchmark.ipynb` for the Epochal Sawtooth Phenomenon (ESP) realized by using incremential quadratic optimization.

Fig. 19 The effects of data shuffling on incremental quadratic optimization.
![Fig. 19 The effects of data shuffling on incremental quadratic optimization.](img/replication-benchmark.png)


 - See `replication-example-analysis.ipynb` for the analysis on replication example (using Adam), including:


 Fig. 11 (a) m norm over epoch 4-6; (b) v norm over epoch 4-6. Momentum takes on large value at the beginning of epoch, then drop exponentially; afterwards gradually increases. ∥v∥ steadily increases during epoch.
 ![Fig 11](img/m-v-norms.png)


 Fig. 13 ⟨gt,∇lbt ⟩ for batch b=100.
 ![Fig 13](img/dot-gt-g100.png)


 Fig. 14 (a) ⟨mt,∇lbt ⟩
 ![Fig 14](img/dot-mt-g100.png)


 Fig. 15 (a) ∥gt∥ over epoch 4 and 5; (b) Regression line of ∥gt∥ over epoch 4.
![Fig 15](img/grad_norm.png)


 Fig. 17 (a) ⟨Δθt,∇lb t ⟩ over epoch 4 for b=100
![Fig 17](img/dot-deltax-g100.png)

 - See `replication-example-rmsprop.ipynb` for the analysis on replication example (using RMSProp), including:

Fig. 21 Incremental quadratic optimization replication using RMSProp optimizer.
![Fig 21](img/replication-rmsprop.png)


 - See `replication-example-loss-init.ipynb` for the initial loss:
 Fig. 10 (a) Training loss curve when plot every=1. (b) Histogram of loss of all batches at the start of epoch 4.
![Fig 10](img/loss-all-batches-init.png)


 - See `replication-example-reverse-and-with-replacement.ipynb` for the reversing the sample sequence and sample with replacement:

   
 Fig. 19 The effects of data shuffling on incremental quadratic optimization. (a) Incremental quadratic optimization with shuffle=True. ESP is replicated when we shuffle data. Smaller β2 exacerbates ESP. Similar to Figure 4 (b) When Shuffle=False, ESP is not observed. (c) Reverse the sample sequence for each epoch. This significantly amplifies the ESP, aligning with our earlier analysis illustrated in Figure 12. (d) Sample with replacement. ESP is not observed.
![Fig 19](img/replication-benchmark.png)


 - See `replication-example-analysis-different-beta2_v2.ipynb` and `replication-example-analysis-different-eps.ipynb` for the analysis on replication example (using Adam) under different beta_2 and epsilon values:
Fig. 22 The effects of varying β2 and ϵ
![Fig 22](img/vary_beta2_and_eps.png)


 - See `dot_product_analysis.ipynb` for the 3D example of dot product analysis:
 Fig. 16 The n-shaped similarity explained by low dimensional example
 ![Fig 16](img/dot_deltax_flip_sign_explain.png)


 - Additional result for test on model size:

The results for BERT-tiny. Model size is the significant influencing factor.

 ![additional figure](img/additional_figure_model_size.png)

 - Additional result for test on plot-every parameter:

The parameter plot every defines the number of steps used for averaging when
plotting. When the batch size is fixed, the effect of plot every is shown in Figure below. As plot every increases, the ESP weakens.

 ![additional figure](img/additional_figure_plot_every.png)

 - Additional result for test on batch size:

Studying the effect of batch size is slightly more complex. To ensure that the number of samples used for averaging (i.e.,
batch size * plot every) remains the same, we adjust the plot every parameter accordingly. The results are displayed in Figure below. As batch size increases, ESP also
weakens and become less discernible.

 ![additional figure](img/additional_figure_batch_size.png)

 - Additional result for test on weight decay:

The weight decay parameter has a relatively small impact on ESP. Larger values of weight decay slightly weaken the ESP, as shown in Figure below. Model size, however, has a significant impact on ESP, as shown in Figure 9. As the model size increases, ESP becomes more prominent.

 ![additional figure](img/additional_figure_weight_decay.png)

 - Additional result for test on beta_1:

Effects of β1 settings on ESP; (a) BERT-tiny with betas=(0.5, 0.999); (b) BERT-tiny
with betas=(0, 0.999). β1 does not have a noticeable effect on ESP.

 ![additional figure](img/additional_figure_beta_1.png)

 - Additional result for test on batch size for replication example:

(a) batch size=100; (b) batch size=200. Larger batch size can mitigates ESP.

 ![additional figure](img/additional_figure_replication_batch_size.png)

 - Additional result for test on betas for replication example:

(a) β1 = 0.9; (b) β1 = 0.1; ESP becomes less pronounced when β1 get smaller; this feature
differs from the BERT example (Figure 5).

 ![additional figure](img/additional_figure_replication_betas.png)

 - Additional result for test on rmsprop optimizor for replication example:

Incremental quadratic optimization replication using RMSProp optimizer. (a)
shuffle=True; ESP still exists, but it is much more subtle. At epoch 10, the training loss increased by
about 10% at the end of the epoch. (b) shuffle=False. No shuffling mitigates ESP, similar to Adam.

 ![additional figure](img/additional_figure_replication_rmsprop.png)

 - Additional result for loss curve comparision:

l^t_t compared with l^b_t for fixed b = 100; they exhibit the same pattern shown by Eqn 10.

 ![additional figure](img/additional_figure_loss_curves_comparison.png)

