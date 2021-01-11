# Description:
# Module used to display games between AI's and to play vs AI

# Imports:
import pygame
import numpy as np
import random
import game_mechanics as gm
import data_functions as df
import model_new as model

board_len = 30


def drawgrid(w, rows, surface):
    sizeBtwn = w//rows
    x = 0
    y = 0
    for l in range(rows + 1):
        pygame.draw.line(surface, (0, 0, 0), (x, 0), (x, w))
        pygame.draw.line(surface, (0, 0, 0), (0, y), (w, y))
        x += sizeBtwn
        y += sizeBtwn


def display_moves(x_moves, o_moves, surface, textsurfaceX, textsurfaceO, x_random=None, o_random=None, textsurfaceX_random=None, textsurfaceO_random=None, show_randomness=False):
    for index, move in enumerate(x_moves):
        if show_randomness is True and x_random[index] is True:
            surface.blit(textsurfaceX_random, (move[1], move[0]))
        else:
            surface.blit(textsurfaceX, (move[1], move[0]))
    for index, move in enumerate(o_moves):
        if show_randomness is True and o_random[index] is True:
            surface.blit(textsurfaceO_random, (move[1], move[0]))
        else:
            surface.blit(textsurfaceO, (move[1], move[0]))


def play_game(mode, memory=None, speed=None, show_randomness=False, player_symbol=None, model_name=None, return_history=False):
    # Mode can be: 'render', 'render_outcomes', 'pvp', 'pva'
    pygame.init()

    value_test_model = model.AlphaTTT('X', value_name='val_test1')

    if mode == 'pva':
        if player_symbol == 'X':
            agent_symbol = 'O'
        else:
            agent_symbol = 'X'
        ttt_agent = model.AlphaTTT(agent_symbol, policy_name=model_name)

    # CONSTANT VARIABLES
    # ==================================
    screen_width = 901
    first_square = (4, -6) # add 30 to each cordinate for next one
    history = []

    main_window = pygame.display.set_mode((screen_width, screen_width))
    pygame.display.set_caption("Tic-Tac-Toe")
    main_window.fill((192, 192, 192))

    # Timing:
    clock = pygame.time.Clock()
    current_time = 0
    move_placed_time = 0
    render_time = 0

    myfont = pygame.font.SysFont('courier new', 40)
    textsurfaceX = myfont.render('X', False, (0, 0, 0))
    textsurfaceX_random = myfont.render('X', False, (255, 0, 0))
    textsurfaceO = myfont.render('O', False, (0, 0, 0))
    textsurfaceO_random = myfont.render('O', False, (255, 0, 0))

    pygame.display.flip()

    player = 'X'
    turn = 1
    x_moves = []
    x_random = []
    o_moves = []
    o_random = []

    run = True
    clear_board = False
    abort = False
    game_board = gm.create_board()
    while run is True:
        main_window.fill((192, 192, 192))
        drawgrid(screen_width, 30, main_window)

        (mouse_x, mouse_y) = pygame.mouse.get_pos()
        mouse_x = int(mouse_x/30)
        mouse_y = int(mouse_y/30)

        pygame.draw.rect(main_window, (105, 105, 105), (1 + 30*mouse_x, 1 + 30*mouse_y, 29, 29))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                abort = True
            if event.type == pygame.MOUSEBUTTONDOWN and ((first_square[1] + 30*mouse_y, first_square[0]+30*mouse_x) not in x_moves + o_moves):

                data = df.get_prediction_format(game_board, player)
                pred = value_test_model.value_network(data)
                print(pred)

                if player == 'X' and turn == 1 and mode == 'pvp':
                    x_moves.append((first_square[1] + 30*mouse_y, first_square[0]+30*mouse_x))
                    if return_history is True:
                        history.append([mouse_y, mouse_x])
                    gm.select_space(game_board, player, (mouse_y, mouse_x))
                    turn *= -1
                    player = 'O'
                elif player == 'O' and turn == -1 and mode == 'pvp':
                    o_moves.append((first_square[1] + 30 * mouse_y, first_square[0] + 30 * mouse_x))
                    if return_history is True:
                        history.append([mouse_y, mouse_x])
                    gm.select_space(game_board, player, (mouse_y, mouse_x))
                    turn *= -1
                    player = 'X'
                elif mode == 'pva' and player_symbol == 'X' and turn == 1:
                    x_moves.append((first_square[1] + 30*mouse_y, first_square[0]+30*mouse_x))
                    gm.select_space(game_board, player_symbol, (mouse_y, mouse_x))
                    turn *= -1
                elif mode == 'pva' and player_symbol == 'O' and turn == -1:
                    o_moves.append((first_square[1] + 30 * mouse_y, first_square[0] + 30 * mouse_x))
                    gm.select_space(game_board, player_symbol, (mouse_y, mouse_x))
                    turn *= -1

        current_time = pygame.time.get_ticks()

        if mode == 'render':
            if len(memory) > 0:
                player = memory[0][0]
                move = memory[0][1]

                randomness = memory[0][2]
                delta_move_time = current_time - move_placed_time
                if player == 'X' and delta_move_time > speed:
                    x_moves.append((first_square[1] + 30*move[1], first_square[0] + 30*move[0]))
                    x_random.append(randomness)
                    gm.select_space(game_board, player, move)
                    move_placed_time = pygame.time.get_ticks()
                    memory.pop(0)
                elif player == 'O' and delta_move_time > speed:
                    o_moves.append((first_square[1] + 30*move[1], first_square[0] + 30*move[0]))
                    o_random.append(randomness)
                    gm.select_space(game_board, player, move)
                    move_placed_time = pygame.time.get_ticks()
                    memory.pop(0)

        if mode == 'render_outcomes':
            # Renders just end_states of games, game histories will be in memory variable

            delta_render_time = current_time - render_time

            if delta_render_time > speed:
                game_hist = memory[0]
                x_moves = []
                o_moves = []
                for i, move in enumerate(game_hist):
                    if i % 2 == 0:
                        x_moves.append((first_square[1] + 30*move[0], first_square[0] + 30*move[1]))
                    else:
                        o_moves.append((first_square[1] + 30*move[0], first_square[0] + 30*move[1]))

                memory.pop(0)
                render_time = pygame.time.get_ticks()
                if len(memory) == 0:
                    run = False

        if mode == 'pva':
            if agent_symbol == 'X' and turn == 1:
                # ttt_agent.memorize_state(game_board.copy())
                move, randomness = ttt_agent.policy_predict_action(game_board, first_move=(len(x_moves) + len(o_moves) == 0), epsilon=0)
                x_moves.append((first_square[1] + 30*move[0], first_square[0] + 30*move[1]))
                gm.select_space(game_board, agent_symbol, move)
                turn *= -1
            elif agent_symbol == 'O' and turn == -1:
                # ttt_agent.memorize_state(game_board.copy())
                move, randomness = ttt_agent.policy_predict_action(game_board, 0)
                o_moves.append((first_square[1] + 30 * move[0], first_square[0] + 30 * move[1]))
                gm.select_space(game_board, agent_symbol, move)
                turn *= -1

        display_moves(x_moves, o_moves, main_window, textsurfaceX, textsurfaceO, x_random=x_random, o_random=o_random, textsurfaceX_random=textsurfaceX_random, textsurfaceO_random=textsurfaceO_random, show_randomness=show_randomness)
        pygame.display.update()
        clock.tick(60)

        if abort is False:
            run = (gm.game_over(game_board) == -1)

        if clear_board:
            game_board = gm.create_board()
            x_moves = []
            o_moves = []
            clear_board = False

    pygame.quit()

    if return_history is True:
        return np.array(history)
    else:
        return

