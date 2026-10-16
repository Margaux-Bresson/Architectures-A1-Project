# --- Étape 1 : build de l'application Vite ---
FROM node:22-alpine AS build

WORKDIR /app

# Dépendances d'abord (couche mise en cache tant que package-lock.json ne change pas)
COPY package.json package-lock.json ./
RUN npm ci

# URL de l'API, intégrée au bundle au moment du build (lue via import.meta.env.VITE_API_URL)
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY . .
RUN npm run build

# --- Étape 2 : service des fichiers statiques avec nginx ---
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]