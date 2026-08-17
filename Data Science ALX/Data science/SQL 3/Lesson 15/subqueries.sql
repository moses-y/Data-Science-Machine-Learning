SELECT
	country_name,
    AVG(est_gdp_in_billions) as Avg_GDP,
    AVG(est_population_in_millions) AS Avg_population
FROM
	(SELECT
		country_name,
		est_gdp_in_billions,
		est_population_in_millions
	FROM
		united_nations.Economic_indicators
	WHERE
		pct_unemployment > 5
		AND Time_period = 2020) as FilteredCountries
GROUP BY
	country_name
