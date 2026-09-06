const app = require('./app');
const config = require('./config/env');
const { connectDB, disconnectDB } = require('./config/db');

const startServer = async () => {
  // Connect to MongoDB
  await connectDB();

  // Start HTTP Server
  const server = app.listen(config.port, () => {
    console.log(`=========================================`);
    console.log(` TruthLens AI - Backend Service`);
    console.log(` Environment : ${config.nodeEnv}`);
    console.log(` Server URL  : http://localhost:${config.port}`);
    console.log(` ML Service  : ${config.mlServiceUrl}`);
    console.log(` MongoDB     : ${config.mongoUri}`);
    console.log(`=========================================`);
  });

  // Graceful shutdown handling
  const shutdown = async (signal) => {
    console.log(`\n[Server] Received ${signal}. Gracefully shutting down...`);
    server.close(async () => {
      console.log('[Server] HTTP server closed.');
      await disconnectDB();
      process.exit(0);
    });

    // Force exit if shutdown hangs
    setTimeout(() => {
      console.error('[Server] Forced shutdown after timeout.');
      process.exit(1);
    }, 10000);
  };

  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
};

startServer();
