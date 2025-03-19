require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const moment = require("moment");
const app = express();
app.use(cors());

// const PORT = process.env.PORT || 5000;
const API_KEY_NEWS = process.env.news_apikey; 
app.get("/news/:company", async (req, res) => {
    const company = req.params.company.toLowerCase();
    
    // Get dates for past week
    const oneWeekAgo = moment().subtract(7, "days").format("YYYY-MM-DD");
    const today = moment().format("YYYY-MM-DD");

    try {
        const response = await axios.get(
            `https://newsapi.org/v2/everything?q=${encodeURIComponent(company)}&from=${oneWeekAgo}&to=${today}&sortBy=publishedAt&apiKey=${API_KEY_NEWS}`
        );

        const filteredNews = response.data.articles.filter(article => {
            const title = article.title ? article.title.toLowerCase() : "";
            const description = article.description ? article.description.toLowerCase() : "";
            
            return title.includes(company) || description.includes(company);
        });
        console.log("hello bbgl");
        res.json(filteredNews);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});


app.listen(3000, () => console.log(`Server running on port ${3000}`));