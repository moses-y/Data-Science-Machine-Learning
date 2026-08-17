SELECT COUNT(*)
FROM well_pollution
WHERE description LIKE 'Clean_%' OR (results = 'Clean' AND biological < 0.01);
-- counts and the below returns all the rows

SELECT *
FROM well_pollution
WHERE description LIKE 'Clean_%' OR results = 'Clean' AND biological < 0.01;