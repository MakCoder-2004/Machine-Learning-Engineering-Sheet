async function doTraining(model) {
  const history = await model.fit(xs, ys, {
    epochs: 1000,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch: ${epoch} Loss: ${logs.loss}`);
      },
    },
  });
}

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
model.summary();

const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

doTraining(model).then(() => {
  model.predict(tf.tensor2d([5], [1, 1])).print();
});
