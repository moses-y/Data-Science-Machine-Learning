SELECT
	loc.Country_name,
    eco.Time_period,
    ifnull(eco.pct_unemployment, 19.59) as pct_unemployment_imputed
    
FROM
	united_nations.geographic_location AS loc

LEFT JOIN
	united_nations.Economic_Indicators as eco
    ON eco.country_name = loc.Country_name
    
WHERE REGION LIKE "%Central and Southern Asia%"

UNION

SELECT
	loc.Country_name,
    eco.Time_period,
    ifnull(eco.pct_unemployment, 19.59) as pct_unemployment_imputed
    
FROM
	united_nations.geographic_location AS loc

LEFT JOIN
	united_nations.Economic_Indicators as eco
    ON eco.country_name = loc.Country_name
    
WHERE REGION LIKE "%Ocenia%"

UNION

SELECT
	loc.Country_name,
    eco.Time_period,
    ifnull(eco.pct_unemployment, 19.59) as pct_unemployment_imputed
    
FROM
	united_nations.geographic_location AS loc

LEFT JOIN
	united_nations.Economic_Indicators as eco
    ON eco.country_name = loc.Country_name
    
WHERE REGION LIKE "%Subsaharan Africa%";

