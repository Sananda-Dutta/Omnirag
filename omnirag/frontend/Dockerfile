# Vite + React + TypeScript app (build output in /app/dist).
# The app calls the API with a relative URL (/api/v1), so no API URL is
# baked in at build time. Caddy routes /api/* to the backend.
 
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
 
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
 