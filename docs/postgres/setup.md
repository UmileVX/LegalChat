# PostgreSQL setup

## Create user

```sql
-- create normal user
CREATE USER username WITH PASSWORD 'your_password';

-- create super user
CREATE USER username WITH PASSWORD 'your_password' SUPERUSER;
```

By using the query below, you could check all created users (use it with admin user):
```sql
SELECT * FROM pg_user WHERE usename = 'your_new_username';
```

### Grouping users with roles

PostgreSQL roles facilitate easier user management by enabling administrators to group users according to shared responsibilities or characteristics.
This goes beyond the fundamental difference between normal users and superusers.
An adaptable and scalable method of managing user access is through roles.

Administrators can effectively manage permissions by creating roles.
For example:
```sql
-- create a role
CREATE ROLE sales_team;

-- grant qualifications to role
GRANT SELECT ON ALL TABLES IN SCHEMA public TO sales_team;

-- assign role to user
GRANT sales_team TO sales_example_user;
```

## Referenes

- [Creating user, database and adding access on PostgreSQL](https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e)
