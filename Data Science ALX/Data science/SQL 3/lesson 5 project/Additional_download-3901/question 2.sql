SELECT source_id, MAX(number_of_people_served) as max_people_served
FROM md_water_services.water_source
GROUP BY source_id
ORDER BY max_people_served DESC
LIMIT 100;
