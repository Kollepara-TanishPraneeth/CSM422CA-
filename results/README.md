# Results Directory

This directory stores evaluation metrics, training history plots, and performance reports generated after training the ReviewSense AI model.

## Genuine Evaluation Policy
In adherence to academic and engineering integrity, no fabricated metrics or fake training curves are stored here. Results are generated dynamically upon actual model execution.

## Expected Outputs

Once `train.py` or `notebooks/ReviewSenseAI_Training.ipynb` finishes execution, the following files will be created here:

1. **`training_metrics.png`**:
   - Matplotlib visualization containing:
     - Training vs. Validation Loss across epochs.
     - Training vs. Validation Accuracy across epochs.
2. **`evaluation_report.json`**:
   - Final test loss and test accuracy computed on the 25,000 unseen IMDB test reviews.

## How to Generate Results
Run either:
```bash
python train.py
```
Or execute all cells in `notebooks/ReviewSenseAI_Training.ipynb` on Google Colab and download the generated plot to this directory.
