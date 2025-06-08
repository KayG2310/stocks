const { spawn } = require('child_process');
const path = require('path');

/**
 * Calls a Python script and returns its output.
 * @param {string} scriptPath - The relative path to the Python script.
 * @param {Array<string>} args - Arguments to pass to the Python script.
 * @returns {Promise<string>} - Resolves with Python script output.
 */
function callPython(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const fullPath = path.resolve(scriptPath); // Ensure absolute path
    const py = spawn('python3', [fullPath, ...args]);

    let data = '';
    let error = '';

    py.stdout.on('data', (chunk) => {
      data += chunk.toString();
    });

    py.stderr.on('data', (chunk) => {
      error += chunk.toString();
    });

    py.on('close', (code) => {
      if (code !== 0) {
        reject(`Python process exited with code ${code}: ${error}`);
      } else {
        resolve(data.trim());
      }
    });
  });
}

// Example usage (can be removed in production):
(async () => {
  try {
    const ticker = 'AAPL'; // example ticker
    const result = await callPython('backend/python/predict.py', ["TSLA"]);
    const json = JSON.parse(result);
    console.log('Predicted Sentiment:', json);
  } catch (err) {
    console.error('Error:', err);
  }
})();

module.exports = { callPython };
