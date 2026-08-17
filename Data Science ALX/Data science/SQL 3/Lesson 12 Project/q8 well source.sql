SELECT ROUND(AVG(number_of_people_served)) AS AveragePeoplePerWell
FROM md_water_services.water_source
WHERE type_of_water_source = 'Well';
