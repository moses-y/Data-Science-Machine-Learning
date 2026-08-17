SELECT
	econ.country_name,
    econ.time_period,
    econ.est_gdp_in_billions,
    service.pct_managed_drinking_water_services
    
FROM
	united_nations.economic_indicators as econ
    
INNER JOIN
	united_nations.basic_services as service

ON
	econ.country_name = service.country_name
    AND econ.time_period = service.time_period
    
WHERE
	econ.time_period = 2020
    AND service.pct_managed_drinking_water_services < 90
    AND econ.est_gdp_in_billions > (
									SELECT
										AVG(Est_gdp_in_billions)
									FROM
										united_nations.Economic_Indicators
									WHERE
										Time_period = 2020);
    
    