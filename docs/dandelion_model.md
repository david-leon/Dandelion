## U-net FCN
Implementation of U-net FCN

```python
class model_Unet(channel=1, im_height=128, im_width=128, Nclass=2, kernel_size=3, border_mode='same', base_n_filters=64, output_activation=softmax)
```
* **channel**: input channel number
* **Nclass**: output channel number

The model accepts input of shape in the order of (B, C, H, W), and outputs with shape in the order of (B, H, W, C).
