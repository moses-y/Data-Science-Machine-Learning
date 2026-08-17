SELECT 
    e.employee_name, 
    COUNT(DISTINCT v.location_id) AS unique_sites_visited
FROM 
    md_water_services.employee AS e
LEFT JOIN 
    md_water_services.visits AS v ON e.assigned_employee_id = v.assigned_employee_id
GROUP BY 
    e.assigned_employee_id, e.employee_name
ORDER BY 
    unique_sites_visited DESC
LIMIT 30;
