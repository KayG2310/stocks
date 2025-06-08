const { spawn } = require('child_process');

/**
 * Calls a Python script and returns its output.
 * @param {string} scriptPath - The path to the Python script.
 * @param {Array<string>} args - Arguments to pass to the Python script.
 * @returns {Promise<string>} - Resolves with Python script output.
 */
function callPython(scriptPath, args = []) {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', [scriptPath, ...args]);

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

// Example usage:
(async () => {
  try {
    const result = await callPython('backend/python/yourScript.py', ['arg1', 'arg2']);
    console.log('Output from Python:', result);
  } catch (err) {
    console.error('Error:', err);
  }
})();

module.exports = { callPython };
