# High level wrapper for standarscaling
Light wrapper for easier standardscaling when using PyTorch  
Scaler expects data to be in shape (n_samples, timesteps, n_features) and expects you to want the scaling to be done like this:

```python
data = data.reshape(-1, n_features)
data = scaler.transform(data)
data = data.reshape(-1, timesteps, n_features)
```


# Useage

### Fit from torch loader
```python
ts = TorchScaler()
ts.fit_from_loader(train_loader)
```

#### Transforming
```python
data = ts.transform(data)
```

#### Save
```python
ts.save("my_scaler.pickle")
```

#### Load
```python
ts = TorchScaler("my_scaler.pickle")
```