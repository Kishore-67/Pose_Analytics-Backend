const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu'); 
const { createCanvas, loadImage } = require('canvas');
const poseDetection = require('@tensorflow-models/pose-detection');

const admin = require('firebase-admin');
const serviceAccount = require('./firebase-secret.json'); 

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

let detector;

async function initDetector() {
  await tf.setBackend('cpu');
  await tf.ready();

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
    }
  );

  console.log('Pose detector initialized with CPU backend');
}

app.post('/pose', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file provided' });
  }

  try {
    const imgBuffer = req.file.buffer;
    const img = await loadImage(imgBuffer);

    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const poses = await detector.estimatePoses(canvas);

    if (poses.length === 0) {
      return res.status(400).json({ error: 'No pose detected' });
    }

    const keypoints = poses[0].keypoints;

    const docRef = await db.collection('pose_detections').add({
      timestamp: new Date(),
      keypoints: keypoints
    });

    console.log(`Saved pose detection with ID: ${docRef.id}`);
    
    res.json({
      message: 'Pose detected and saved to Firestore',
      documentId: docRef.id,
      keypoints
    });
    console.log(keypoints);

  } catch (err) {
    console.error('Error processing image:', err);
    res.status(500).json({ error: 'Error processing image' });
  }
});

const PORT = process.env.PORT || 5000;

initDetector().then(() => {
  app.listen(PORT, () => {
    console.log(`Pose backend listening on port ${PORT}`);
  });
});
