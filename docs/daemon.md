# Daemon

By default, when Postgresql starts, pgvecto.rs will start a background worker process to handle index creation and queries. However, you can move the worker process to machines connected with network.

Assume you have downloaded the source code,

```bash
cargo build --package service --bin daemon --release
```

It will generate the daemon executable binary. You can find it in `./target/release`.

Now you can run the daemon on port `9999` and data directory `data`.

```bash
daemon --addr 0.0.0.0:9999 --chdir ./data
```

Connect to Postgresql and set the address of remote daemon.

```sql
ALTER SYSTEM SET vectors.remote = '0.0.0.0:9999';
```

Restart Postgresql, and all index creation and queries will be handled by the remote daemon process.
