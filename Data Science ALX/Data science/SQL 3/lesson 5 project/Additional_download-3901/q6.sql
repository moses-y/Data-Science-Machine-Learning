SELECT * FROM md_water_services.employee WHERE position = 'Field Surveyor'
AND (phone_number LIKE '%86%' OR phone_number LIKE '%11%')
AND (employee_name LIKE '% A%' OR employee_name LIKE '% M%');
