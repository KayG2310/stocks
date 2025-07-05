// backend/node/models/StockTickerMapping.js
const mongoose = require('mongoose');

const StockTickerMappingSchema = new mongoose.Schema({
  StockSymbol: {
    type: String,
    required: true,
    unique: true
  },
  CommonName: {
    type: String,
    required: true
  }
});

module.exports = mongoose.model('StockTickerMapping', StockTickerMappingSchema);
