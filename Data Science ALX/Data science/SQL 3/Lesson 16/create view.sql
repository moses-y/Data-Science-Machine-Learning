CREATE VIEW united_nations.country_unemployment_rate
as
SELECT
	loc.country_name,
    eco.Time_period,
    eco.pct_unemployment
    
FROM
	united_nations.geographic_location as loc
    
LEFT JOIN
	united_nations.economic_indicators as eco
        
ON eco.country_name = loc.country_name
WHERE Region = 'sub-saharan africa';