import { MnistData } from "./data.js";
var canvas, ctx, saveButton, clearButton;
var pos = { x: 0, y: 0 };
var rawImage;
var model;

// create the model
function getModel() {
  model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 3,
      filters: 8,
      activation: "relu",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(
    tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu" }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// train the model using the data
async function train(model, data) {
  const metrics = ["loss", "val_loss", "accuracy", "val_accuracy"];
  const container = { name: "Model Training", styles: { height: "640px" } };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 20,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

// update the position of the mouse on the canvas
function setPosition(e) {
  pos.x = e.clientX - 100;
  pos.y = e.clientY - 100;
}

// draw a line on the canvas
function draw(e) {
  if (e.buttons != 1) return;
  ctx.beginPath();
  ctx.lineWidth = 24;
  ctx.lineCap = "round";
  ctx.strokeStyle = "white";
  ctx.moveTo(pos.x, pos.y);
  setPosition(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  rawImage.src = canvas.toDataURL("image/png");
}

// clear the canvas
function erase() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
}

// convert the image to a tensor and predict the class
function save() {
  // Use canvas directly instead of rawImage to avoid empty image errors
  var raw = tf.browser.fromPixels(canvas, 1);
  var resized = tf.image.resizeBilinear(raw, [28, 28]);
  var tensor = resized.expandDims(0);
  var prediction = model.predict(tensor);
  var pIndex = tf.argMax(prediction, 1).dataSync();

  alert("Predicted Class: " + pIndex[0]);
}

// handle image upload
function handleUpload(e) {
  var file = e.target.files[0];
  if (!file) return;

  var reader = new FileReader();
  reader.onload = function (event) {
    var img = new Image();
    img.onload = function () {
      // Clear canvas with black background (expected by the model)
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, 280, 280);
      // Draw uploaded image
      ctx.drawImage(img, 0, 0, 280, 280);
      rawImage.src = canvas.toDataURL("image/png");
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
}

// initialize the application
function init() {
  canvas = document.getElementById("canvas");
  rawImage = document.getElementById("canvasimg");
  ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 280, 280);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mousedown", setPosition);
  canvas.addEventListener("mouseenter", setPosition);
  saveButton = document.getElementById("sb");
  saveButton.addEventListener("click", save);
  clearButton = document.getElementById("cb");
  clearButton.addEventListener("click", erase);
  document
    .getElementById("imageUpload")
    .addEventListener("change", handleUpload);
}

// main
async function run() {
  const data = new MnistData();
  await data.load();
  const model = getModel();
  tfvis.show.modelSummary({ name: "Model Architecture" }, model);
  await train(model, data);
  init();
  alert("Training is done, try classifying your handwriting!");
}

document.addEventListener("DOMContentLoaded", run);
