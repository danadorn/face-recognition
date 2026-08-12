# Face Recognition

This project contains a Face Verification System built with:
- **Haar Cascade** for face detection
- **ArcFace ResNet100** for face verification

## Setup

1. **Download the ArcFace Model**

   Download the ArcFace ResNet100 ONNX model and place it in the `models/` directory:
   ```
   models/arcfaceresnet100-8.onnx
   ```

   You can download it from:
   - [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/arcface-int8)
   - Or any ONNX model repository

2. **Install Dependencies Locally (Optional)**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Locally**

   ```bash
   python app.py
   ```

   The app will be available at `http://localhost:7860`

## Deployment to Vercel

### Using Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```


## Important Notes

- The ArcFace model file (`arcfaceresnet100-8.onnx`) must be included in your repository or downloaded during build
- The Haar Cascade classifier is auto-downloaded from OpenCV's GitHub repository
- Vercel has limitations on execution time (up to 60 seconds for serverless functions), so very large models may timeout
- For persistent hosting with long running processes, consider alternatives like Heroku, Railway, or Hugging Face Spaces

## File Structure

```
.
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel configuration
├── models/
│   └── arcfaceresnet100-8.onnx    # ArcFace model (download separately)
└── README.md
```

## Alternative: Use Gradio Cloud (Easiest)

Instead of Vercel, you can use Gradio Cloud:

```python
# In app.py, change the last line from:
# demo.launch()
# To:
demo.launch(share=True)  # Creates a shareable link
# Or for persistent hosting:
# demo.launch(share=False)  # Run locally and share via CLI
```

Then run:
```bash
python app.py
```
