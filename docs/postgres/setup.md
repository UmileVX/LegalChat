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

## List all of the tables in schema

```sql
-- list all tables of all schemas
\dt *.*

-- list all tables of public schema
\dt public.*
```

## Install PostgreSQL on Ubuntu 24.04

1. Install the postgresql-common dependency package on your server.

```bash
sudo apt install -y postgresql-common -y
```

2. Run the following command to execute the PostgreSQL APT repository script.

```bash
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
```

3.Install the postgresql database server package.

```bash
sudo apt install -y postgresql
```

3-1. Install `postgresql-server-dev` package (replace xx with your postgres version)

```bash
sudo apt install postgresql-server-dev-XX

# for postgresql@17:
sudo apt install postgresql-server-dev-17
```

4. Start the DB service

```bash
sudo systemctl restart postgresql
```

## Remote access

To allow remote access, we first need to modify the `postgresql.conf` file.

```bash
cd /etc/postgresql/17/main

sudo vi postgresql.conf
```

Then modify the `listen_addresses` as following:
```
listen_addresses = '*'
```

Next, edit the `pg_hba.conf` file to allow the `SSL` features for the target client IP:
```bash
sudo vi pg_hba.conf
```

Add the following lines to allow the client (222.112.65.155) to connect (assume that 222.112.65.155 is the IP address that we want to grant access):
```
# Allow SSL connections
host    all             all             222.112.65.155/32    scram-sha-256

# Allow non-SSL connections (optional, but use with caution)
host    all             all             222.112.65.155/32    md5
```
Replace scram-sha-256 with md5 if your PostgreSQL version doesn't support SCRAM authentication or if your setup requires it.

If you want to allow all IPs, then put the following:
```
host    all             all             0.0.0.0/0            scram-sha-256
```

Finally, reload the postgresql with `sudo systemctl reload postgresql`.

## Referenes

- [Creating user, database and adding access on PostgreSQL](https://medium.com/coding-blocks/creating-user-database-and-adding-access-on-postgresql-8bfcd2f4a91e)
- [How to install postgresql on Ubuntu](https://docs.vultr.com/how-to-install-postgresql-on-ubuntu-24-04)
- [Install postgresql-server-dev package](https://stackoverflow.com/a/61877103/9012940)
