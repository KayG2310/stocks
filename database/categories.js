const mongoose = require('mongoose');
const Schema = mongoose.Schema;
mongoose.connect(`mongodb://localhost:27017/stocknews`);
const categorySchema = new Schema({
    FullName: {
        type: String,
        required: true
    },
    CommonName: {
        type: String,
        required: true
    },
    StockSymbol: {
        type: String,
        required: true
    },
    Sector:{
        type: String,
        required: true
    }
});

module.exports = mongoose.model('categories', categorySchema);